# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dataset-tools is a collection of Python scripts for normalizing and processing image datasets for machine learning. The tools are designed to work independently as command-line utilities, each focused on a specific dataset processing task.

## Installation & Setup

```bash
pip install -r requirements.txt
```

**Important**: On macOS, it's recommended to install alongside Anaconda due to OpenCV dependencies.

## Common Commands

### Running the main dataset processing tool
```bash
python dataset-tools.py --input_folder path/to/input/ --output_folder path/to/output/ --process_type resize --max_size 512
```

### Running deduplication
```bash
python dedupe.py --input_folder path/to/input/ --output_folder path/to/output/
```

### Running multi-crop tool
```bash
python multicrop.py --input_folder path/to/input/ --output_folder path/to/output/ --min_size 1024
```

### Sorting by image dimensions
```bash
python sort.py --input_folder path/to/input/ --output_folder path/to/output/ --process_type exclude --min_size 1024
```

### Sorting by color
```bash
python sort-color.py --input_folder path/to/input/ --output_folder path/to/output/ --threshold 40
```

### Face detection sorting
```bash
python facesort.py --input_folder path/to/input/ --output_folder path/to/output/ --method faces
```

### Object detection-based cropping
```bash
python obj_detect_cropper.py --input_folder path/to/input/ --output_folder path/to/output/ --bounds_file_path path/to/bounds.csv --file_format runway_csv
```

### OpenPose face cropping
```bash
python openpose_face_cropper.py --input_folder path/to/input/ --output_folder path/to/output/
```

## Architecture & Code Structure

### Independent Script Architecture
This repository follows a flat, script-based architecture where each `.py` file is a standalone tool. There is no central module or package structure - each script can be run independently via command line.

### Common Patterns Across Scripts

All scripts follow similar conventions:

1. **Argument Parsing**: Each script uses `argparse` with `parse_args()` function defining command-line arguments
2. **Main Execution**: Scripts use `if __name__ == "__main__": main()` pattern
3. **Input/Output**: Standard `--input_folder` and `--output_folder` arguments (defaults: `./input/` and `./output/`)
4. **File Format Options**: Most scripts support `--file_extension` flag for `png` or `jpg` output
5. **Verbose Mode**: Most scripts include `--verbose` flag for console progress output

### Image Processing with OpenCV

All scripts use OpenCV (`cv2`) as the primary image processing library. Key patterns:

- Images are loaded with `cv2.imread(file_path)`
- Image validity is checked with `hasattr(img, 'copy')` before processing
- Images are saved via `saveImage()` helper functions with compression settings:
  - PNG: `[cv2.IMWRITE_PNG_COMPRESSION, 0]` (no compression)
  - JPG: `[cv2.IMWRITE_JPEG_QUALITY, 90]`
- Interpolation defaults to `cv2.INTER_CUBIC` for resizing operations

### File System Walking

Scripts use `os.walk()` to recursively process directories:
```python
for root, subdirs, files in os.walk(args.input_folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        # process image
```

### dataset-tools.py Process Types

The main `dataset-tools.py` script supports multiple `--process_type` options:
- `resize`: Resize images to max dimension (default)
- `square`: Make images square by adding borders
- `crop`: Crop to specific dimensions (use with `--height` and `--width`)
- `crop_to_square`: Crop to square by removing edges
- `canny`: Apply Canny edge detection
- `canny-pix2pix`: Create pix2pix paired images with Canny edges
- `scale`: Scale by a factor (use with `--scale`)
- `crop_square_patch`: Random square crop
- `many_squares`: Multiple square crops from one image
- `distance`: Distance transform processing

### Augmentation Support

Many scripts support augmentation flags:
- `--mirror`: Creates horizontally flipped versions
- `--rotate`: Creates 180-degree rotated versions

These are applied via `flipImage()` and `rotateImage()` helper functions after the main processing.

### Object Detection Integration

The `obj_detect_cropper.py` script integrates with external object detection tools:
- **Runway CSV format**: Expects CSV with bounding box coordinates
- **YOLOv5 format**: Expects .txt files with normalized coordinates
- Supports confidence thresholding via `--min_confidence`
- Can crop raw bounding boxes or expand to squares

### Utility Functions

The `utils/load_images.py` module provides multi-threaded image loading:
- Uses threading for parallel image loading
- Thread-safe queue-based architecture
- Useful for loading large datasets efficiently

### Auto-Documentation System

The repository includes an auto-documentation workflow:
- `.github/workflows/update-docs.yml`: GitHub Action that runs on push to main
- `.github/scripts/generate-docs.py`: Generates `docs.md` by running `--help` on all `.py` files
- `docs.md`: Auto-generated, should not be manually edited

## Key Dependencies

- `opencv-python>=4.1.0.25`: Core image processing
- `numpy>=1.7.0`: Numerical operations
- `scipy`: Distance transforms and scientific computing
- `imutils`: Rotation and image manipulation utilities
- `lpips`: Perceptual similarity metrics (used in dedupe.py)
- `scikit-learn` and `scikit-image`: Machine learning and advanced image processing
- `PyMuPDF`: PDF image extraction
- `psd-tools3`: PSD file support
- `mac-tag`: macOS file tagging (macOS only)

## Important Notes for Development

### When Adding New Scripts

1. Follow the established naming convention: lowercase with hyphens or underscores
2. Include standard `parse_args()` function with argparse
3. Support `--input_folder`, `--output_folder`, and `--verbose` at minimum
4. Use the common `saveImage()` pattern for file output
5. Add `--help` support so the script appears in auto-generated docs

### Border and Padding Operations

When working with border operations in `dataset-tools.py`:
- Border types: `stretch`, `reflect`, `solid`, `inpaint`
- Solid borders require `--border_color` in BGR format (e.g., `255,0,0` for blue)
- Division handling for centering is complex - check existing patterns in `makeSquare()` function

### Process Type Implementation

Each process type in `dataset-tools.py` has its own function:
- `makeResize()`: Resize operations
- `makeSquare()`: Square with borders
- `makeSquareCrop()`: Square by cropping
- `makeCanny()`: Canny edge detection
- `makeCrop()`: Arbitrary dimension crops

Output directories are automatically created with naming pattern: `{output_folder}/{type}-{size}/`

### File Extension Handling

When saving images, always split the original filename and replace the extension:
```python
new_file = os.path.splitext(filename)[0] + ".png"
```

This ensures consistency across different input formats.
