#!/usr/bin/env python3
"""
Ken Burns Effect Over Multiple Images

This script implements Ken Burns effects (pan, tilt, zoom) on a sequence of images
captured from the same scene at short intervals to simulate camera movement.

The script processes image ranges defined by start and end frame numbers and applies
smooth transitions using linear interpolation between defined parameters.

Author: Generated for Ken Burns Effect Project
License: GPL-3.0
"""

import os
import sys
import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from wand.image import Image as WandImage
    from wand.color import Color
    from wand import resource
except ImportError:
    print("Error: Wand library not found. Please install it with: pip install Wand")
    sys.exit(1)


# Configuration constants
DEFAULT_PARAMS_FILE = "0_params.txt"
OUTPUT_DIR_PREFIX = "kb-"
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
# Optimal worker limit based on empirical testing - 6 workers max to prevent ImageMagick cache exhaustion
# Beyond 6 workers, ImageMagick runs out of cache resources even with generous limits
MAX_WORKERS = min(1, os.cpu_count() or 1)

# Predefined output sizes
OUTPUT_SIZES = {
    'HD': (1920, 1080),
    '4K': (3840, 2160),
    '8K': (7680, 4320),
}


@dataclass
class CropParameters:
    """Data class to hold crop parameters for a single frame."""
    left_top_x: int
    left_top_y: int
    crop_width: int
    crop_height: int


@dataclass
class EffectParameters:
    """Data class to hold effect parameters from parametric line."""
    first_img: int
    last_img: int
    output_size: Tuple[int, int]
    lefttop_start: Tuple[int, int]
    lefttop_end: Tuple[int, int]
    crop_size_begin: Tuple[int, int]
    crop_size_end: Tuple[int, int]
    pan_step: float = 0.0  # To be calculated per frame
    tilt_step: float = 0.0  # To be calculated per frame


class KenBurnsProcessor:
    """Main class for processing Ken Burns effects on image sequences."""
    
    def __init__(self, input_path: str, params_file: str = DEFAULT_PARAMS_FILE):
        """
        Initialize the Ken Burns processor.
        
        Args:
            input_path: Path to the directory containing input images and parameters file
            params_file: Name of the parameters file
        """
        self.input_path = Path(input_path)
        self.params_file = self.input_path / params_file
        
        # Configure ImageMagick resources based on CPU count and expected workload
        self._configure_imagemagick_resources()
        
        # Validate paths
        if not self.params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {self.params_file}")
        
        # Get sorted list of image files
        self.image_files = self._get_image_files()
        if not self.image_files:
            raise ValueError(f"No supported image files found in {self.input_path}")
        
        print(f"Found {len(self.image_files)} image files")
    
    def _configure_imagemagick_resources(self):
        """Configure ImageMagick resource limits based on system capabilities and thread count."""
        cpu_count = os.cpu_count() or 1
        
        # Calculate resource limits based on CPU count and expected concurrent operations
        # Each 4K image can use ~200-300MB when processed, so we need much more memory
        base_memory_mb = 1024  # Increase base memory per operation
        memory_per_thread_mb = base_memory_mb * cpu_count
        
        # Set ImageMagick resource limits
        try:
            # Memory limit: Scale with CPU count, minimum 4GB, maximum 16GB
            memory_limit_mb = max(4096, min(16384, memory_per_thread_mb))
            memory_limit_bytes = memory_limit_mb * 1024 * 1024
            
            # Map limit: Usually 2x memory limit  
            map_limit_mb = memory_limit_mb * 2
            map_limit_bytes = map_limit_mb * 1024 * 1024
            
            # Disk limit: For temporary files when memory is exceeded
            disk_limit_mb = memory_limit_mb * 4
            disk_limit_bytes = disk_limit_mb * 1024 * 1024
            
            # Thread limit: Limit ImageMagick's internal threading
            thread_limit = max(1, cpu_count // 2)
            
            # Apply the limits
            resource.limits['memory'] = memory_limit_bytes
            resource.limits['map'] = map_limit_bytes
            resource.limits['disk'] = disk_limit_bytes
            resource.limits['thread'] = thread_limit
            
            # Also try setting area limit (width * height * channels)
            # For 4K images: 3840 * 2160 * 4 channels ≈ 33MB per image
            # Allow for multiple images in memory
            area_limit = 3840 * 2160 * 4 * cpu_count * 2  # 2x buffer
            resource.limits['area'] = area_limit
            
            print(f"Configured ImageMagick resources:")
            print(f"  Memory: {memory_limit_mb}MB")
            print(f"  Map: {map_limit_mb}MB") 
            print(f"  Disk: {disk_limit_mb}MB")
            print(f"  Threads: {thread_limit}")
            print(f"  Area: {area_limit // (1024*1024)}MB")
            print(f"  CPU cores: {cpu_count}, Concurrent workers: {cpu_count}")
            
        except Exception as e:
            print(f"Warning: Could not configure ImageMagick resources: {e}")
            print("Continuing with default settings...")
    
    def _get_image_files(self) -> List[Path]:
        """Get alphabetically sorted list of image files in the input directory."""
        image_files = []
        for file_path in self.input_path.iterdir():
            if file_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                image_files.append(file_path)
        
        # Always sort alphabetically for consistent ordering
        image_files.sort()
        
        return image_files
    
    def _parse_output_size(self, size_str: str) -> Tuple[int, int]:
        """
        Parse output size string to width, height tuple.
        
        Args:
            size_str: Size string (e.g., 'HD', '4K', '1280x720')
        
        Returns:
            Tuple of (width, height)
        """
        size_str = size_str.strip()
        
        if size_str in OUTPUT_SIZES:
            return OUTPUT_SIZES[size_str]
        
        # Try to parse custom size (e.g., '1280x720')
        match = re.match(r'(\d+)x(\d+)', size_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        raise ValueError(f"Invalid output size format: {size_str}")
    
    def _parse_coordinate(self, coord_str: str) -> Tuple[int, int]:
        """
        Parse coordinate string to x, y tuple.
        
        Args:
            coord_str: Coordinate string (e.g., '100,200')
        
        Returns:
            Tuple of (x, y)
        """
        coord_str = coord_str.strip()
        try:
            x, y = map(int, coord_str.split(','))
            return x, y
        except ValueError:
            raise ValueError(f"Invalid coordinate format: {coord_str}")
    
    def _parse_size(self, size_str: str) -> Tuple[int, int]:
        """
        Parse size string to width, height tuple.
        
        Args:
            size_str: Size string (e.g., '4K', '1280x720')
        
        Returns:
            Tuple of (width, height)
        """
        return self._parse_output_size(size_str)
    
    def _parse_parameters_line(self, line: str) -> EffectParameters:
        """
        Parse a single parameters line into EffectParameters object.
        
        Args:
            line: Parameters line string
        
        Returns:
            EffectParameters object
        """
        parts = line.strip().split()
        if len(parts) != 7:
            raise ValueError(f"Invalid parameters line format: {line}")
        
        try:
            first_img = int(parts[0])
            last_img = int(parts[1])
            output_size = self._parse_output_size(parts[2])
            lefttop_start = self._parse_coordinate(parts[3])
            lefttop_end = self._parse_coordinate(parts[4])
            crop_size_begin = self._parse_size(parts[5])
            crop_size_end = self._parse_size(parts[6])
            
            # Validate frame numbers
            if last_img <= first_img:
                raise ValueError(f"last_img ({last_img}) must be greater than first_img ({first_img})")
            if first_img < 1 or last_img > len(self.image_files):
                raise ValueError(f"Frame numbers must be between 1 and {len(self.image_files)}")
            
            # Calculate pan and tilt steps per frame
            total_frames = last_img - first_img
            if total_frames > 0:
                pan_step = (lefttop_end[0] - lefttop_start[0]) / total_frames
                tilt_step = (lefttop_end[1] - lefttop_start[1]) / total_frames
            else:
                pan_step = 0.0
                tilt_step = 0.0
            
            return EffectParameters(
                first_img=first_img,
                last_img=last_img,
                output_size=output_size,
                lefttop_start=lefttop_start,
                lefttop_end=lefttop_end,
                crop_size_begin=crop_size_begin,
                crop_size_end=crop_size_end,
                pan_step=pan_step,
                tilt_step=tilt_step
            )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing parameters line '{line}': {e}")
    
    def _load_parameters(self) -> List[EffectParameters]:
        """Load and parse all parameters from the parameters file."""
        parameters = []
        
        with open(self.params_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    params = self._parse_parameters_line(line)
                    parameters.append(params)
                except ValueError as e:
                    print(f"Error on line {line_num}: {e}")
                    continue
        
        if not parameters:
            raise ValueError("No valid parameters found in parameters file")
        
        return parameters
    
    def _interpolate_crop_params(self, params: EffectParameters, frame_num: int) -> CropParameters:
        """
        Calculate crop parameters for a specific frame using linear interpolation.
        
        Args:
            params: Effect parameters
            frame_num: Current frame number (1-based)
        
        Returns:
            CropParameters for the frame
        """
        # Calculate interpolation factor (0.0 to 1.0)
        total_frames = params.last_img - params.first_img
        if total_frames == 0:
            t = 0.0
        else:
            t = (frame_num - params.first_img) / total_frames
        
        # Interpolate position
        start_x, start_y = params.lefttop_start
        end_x, end_y = params.lefttop_end
        pan_step = end_x - start_x
        tilt_step = end_y - start_y

        left_top_x = int(start_x + t * pan_step)
        left_top_y = int(start_y + t * tilt_step)

        # Interpolate crop size
        start_w, start_h = params.crop_size_begin
        end_w, end_h = params.crop_size_end
        
        crop_width = int(start_w + t * (end_w - start_w))
        crop_height = int(start_h + t * (end_h - start_h))
        
        return CropParameters(
            left_top_x=left_top_x,
            left_top_y=left_top_y,
            crop_width=crop_width,
            crop_height=crop_height
        )
    
    def _create_output_directory(self, output_size: Tuple[int, int]) -> Path:
        """Create and return the output directory path (same level as input directory)."""
        # Get the absolute parent directory to ensure proper level placement
        input_abs_path = self.input_path.resolve()
        input_dir_name = input_abs_path.name
        output_dir = input_abs_path.parent / f"{input_dir_name}-kb-processed"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def _process_frame_task(self, frame_num: int, params: EffectParameters, output_dir: Path) -> Tuple[int, bool, str]:
        """
        Process a single frame task for concurrent execution.
        
        Args:
            frame_num: Frame number to process (1-based)
            params: Effect parameters
            output_dir: Output directory path
        
        Returns:
            Tuple of (frame_num, success, error_message)
        """
        try:
            # Get input and output paths
            input_file = self.image_files[frame_num - 1]  # Convert to 0-based index
            output_file = output_dir / f"frame_{frame_num:06d}.jpg"
            
            # Calculate crop parameters for this frame
            crop_params = self._interpolate_crop_params(params, frame_num)
            
            # Process the frame
            self._process_single_frame(
                input_file, output_file, crop_params, params.output_size
            )
            
            return frame_num, True, ""
            
        except Exception as e:
            return frame_num, False, str(e)
    
    def _process_single_frame(self, input_path: Path, output_path: Path, 
                             crop_params: CropParameters, output_size: Tuple[int, int]) -> None:
        """
        Process a single frame with the given crop parameters.
        
        Args:
            input_path: Path to input image
            output_path: Path to output image
            crop_params: Crop parameters for this frame
            output_size: Target output size (width, height)
        """
        try:
            with WandImage(filename=str(input_path)) as img:
                # Ensure crop parameters are within image bounds
                img_width, img_height = img.size
                
                # Adjust crop parameters if they exceed image bounds
                crop_x = max(0, min(crop_params.left_top_x, img_width - 1))
                crop_y = max(0, min(crop_params.left_top_y, img_height - 1))
                crop_w = min(crop_params.crop_width, img_width - crop_x)
                crop_h = min(crop_params.crop_height, img_height - crop_y)
                
                # Crop the image
                img.crop(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
                
                # Resize to output size
                img.resize(output_size[0], output_size[1])
                
                # Save the processed image
                img.save(filename=str(output_path))
                
        except Exception as e:
            raise RuntimeError(f"Error processing frame {input_path}: {e}")
    
    def _process_effect_sequence(self, params: EffectParameters) -> None:
        """
        Process a complete effect sequence according to the given parameters using concurrent processing.
        
        Args:
            params: Effect parameters defining the sequence
        """
        output_dir = self._create_output_directory(params.output_size)
        
        # Create progress bar
        total_frames = params.last_img - params.first_img + 1
        desc = f"Processing frames {params.first_img}-{params.last_img}"
        
        # Prepare frame numbers for processing
        frame_numbers = list(range(params.first_img, params.last_img + 1))
        
        # Process frames concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            with tqdm(total=total_frames, desc=desc, unit="frame") as pbar:
                # Submit all tasks
                future_to_frame = {
                    executor.submit(self._process_frame_task, frame_num, params, output_dir): frame_num
                    for frame_num in frame_numbers
                }
                
                # Process completed tasks
                failed_frames = []
                for future in as_completed(future_to_frame):
                    frame_num, success, error_msg = future.result()
                    
                    if not success:
                        failed_frames.append((frame_num, error_msg))
                    
                    pbar.update(1)
                
                # Report any failures
                if failed_frames:
                    print(f"\n  Warning: {len(failed_frames)} frame(s) failed to process:")
                    for frame_num, error_msg in failed_frames[:5]:  # Show first 5 errors
                        print(f"    Frame {frame_num}: {error_msg}")
                    if len(failed_frames) > 5:
                        print(f"    ... and {len(failed_frames) - 5} more")
    
    def process_all_effects(self) -> None:
        """Process all effect sequences defined in the parameters file."""
        parameters_list = self._load_parameters()
        
        print(f"Loaded {len(parameters_list)} effect sequence(s) from {self.params_file}")
        
        for i, params in enumerate(parameters_list, 1):
            print(f"\nProcessing sequence {i}/{len(parameters_list)}:")
            print(f"  Frames:      {params.first_img} to {params.last_img}")
            print(f"  Output size: {params.output_size[0]}x{params.output_size[1]}")
            print(f"  Pan/tilt:    {params.lefttop_start} → {params.lefttop_end}, pan/tilt steps: {round(params.pan_step, 2)}/{round(params.tilt_step, 2)}")
            print(f"  Zoom:        {params.crop_size_begin} → {params.crop_size_end}")
            
            try:
                self._process_effect_sequence(params)
                print(f"  ✓ Sequence {i} completed successfully")
            except Exception as e:
                print(f"  ✗ Error processing sequence {i}: {e}")
                continue
        
        print("\nAll sequences processed!")


def main():
    """Main function to process Ken Burns effects in current directory."""
    try:
        # Process images in current directory with default parameters file
        processor = KenBurnsProcessor('.', DEFAULT_PARAMS_FILE)
        processor.process_all_effects()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
