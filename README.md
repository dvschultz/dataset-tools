# dataset-tools
Tools for quickly normalizing image datasets for machine learning. I maintain a series of video tutorials on normalizing image datasets—many utilizing this set of scripts—on [my YouTube page](https://www.youtube.com/playlist?list=PLWuCzxqIpJs9v81cWpRC7nm94eTMtohHq).

## Installation
Note: If you’re installing this on a Mac, I highly recommend installing it alongside Anaconda. A video tutorial is available [here](https://www.youtube.com/watch?v=2zgki1oeRkg).
```
git clone https://github.com/dvschultz/dataset-tools.git
cd dataset-tools
pip install -r requirements.txt
```

## Basic Usage
```
python dataset-tools.py --input_folder path/to/input/ --output_folder path/to/output/
```

## Documentation

You can view auto generated documentation in [docs.md](./docs.md)

## Scene Detection for Video

The [scenedetect](./scenedetect/) folder contains a specialized tool for detecting scenes in video footage and extracting them as individual clips. It's optimized for 8mm film footage (both black & white and color) with advanced features for production workflows:

### Key Features

- **Dual-resolution workflow**: Detect scenes on low-resolution versions (480p), apply precise cuts to high-resolution originals (4K)
- **Automatic low-res clip generation**: Creates both high-res and low-res (480p) clips for embeddings and downstream processing
- **ProRes/DNxHD pre-conversion**: Automatically converts high-bitrate formats to H.264 for 10-50x faster cutting with stream copy
- **Three-tier cutting strategy**: Fast stream copy → Accurate stream copy → Re-encode fallback for maximum reliability
- **Automatic timestamp saving**: Saves cut times in JSON and CSV formats for re-cutting original files later
- **Progress tracking with ETA**: Real-time progress indicators and time estimates for long operations
- **Format-specific optimization**: Automatic detection and adaptive processing for B&W vs color footage
- **Smart codec detection**: Auto-detects ProRes, DNxHD, and problematic codecs, adjusting strategy accordingly

### Basic Usage

```bash
# ProRes footage: pre-convert to H.264 for fast cutting (RECOMMENDED for ProRes)
python scenedetect/scene_detect_multiresolution.py \
  -i prores_video.mov \
  -o clips/ \
  --preconvert \
  --bw

# Process directory of videos with auto-detection
python scenedetect/scene_detect_multiresolution.py \
  -d videos_4k/ \
  -o clips/ \
  --auto-detect \
  --adaptive

# Custom low-res output location and save converted files
python scenedetect/scene_detect_multiresolution.py \
  -i video.mp4 \
  -o clips_hires/ \
  -ol clips_480p/ \
  --preconvert \
  --preconvert-dir h264_versions/ \
  --keep-converted
```

### Output Structure

By default, the script creates:
```
clips/                              # High-res clips (or H.264 if pre-converted)
  video_clip_001_bw.mp4
  video_clip_002_bw.mp4
  ...
  video_timestamps.json             # Full metadata (programmatic use)
  video_timestamps.csv              # Human-readable (Excel/Sheets)

clips_lowres/                       # Low-res clips for embeddings
  video_clip_001_bw.mp4             # 480p versions
  video_clip_002_bw.mp4
  ...
```

### Why Pre-Conversion Matters

For ProRes footage at 700+ Mbps:
- **Without pre-conversion**: Re-encodes each of 125 clips individually (~2+ hours)
- **With pre-conversion**: Converts once, then uses fast stream copy (~15 minutes total)

The pre-conversion feature detects ProRes/DNxHD automatically and handles timecode tracks, variable framerate, and other issues that cause stream copy to fail.

### Advanced Options

- `--preconvert`: Auto-convert ProRes/DNxHD to H.264 before cutting
- `--preconvert-dir PATH`: Save converted H.264 files (reuse on subsequent runs)
- `--keep-converted`: Keep H.264 files after processing
- `--bw`: Black and white footage (luma-only detection, higher threshold)
- `--threshold N`: Scene detection sensitivity (28-35 for B&W, 25-30 for color)
- `--adaptive`: Auto-adjust threshold based on footage type
- `--enhance-bw`: Boost contrast for better B&W detection
- `--lowres-height N`: Custom low-res height (default: 480)

For complete documentation, see [scenedetect/README.md](./scenedetect/README.md)

## All Options
### dataset_tools.py
* `--verbose`: Print progress to console.
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--process_type`: Process to use. *Options*: `resize`,`square`,`crop`,`crop_to_square`,`canny`,`canny-pix2pix`,`scale`,`crop_square_patch`,`many_squares`,`distance`  *Default*: `resize`
* `--blur_type`: Blur process to use. Use with `--process_type canny`. *Options*: `none`, `gaussian`, `median`. *Default*: `none`
* `--blur_amount`: Amount of blur to apply (use odd integers only). Use with `--blur_type`. *Default*: `1`
* `--max_size`: Maximum width or height of the output images. *Default*: `512`
* `--force_max`: forces the resize to the max size (by default `--max_size` only scales down)
* `--direction`: Paired Direction. For use with pix2pix process. *Options*: `AtoB`,`BtoA`.  *Default*: `AtoB`
* `--mirror`: Adds mirror augmentation.
* `--rotate`: Adds 90 degree rotation augmentation.
* `--border_type`: Border style to use when using the `square` process type *Options*: `stretch`,`reflect`,`solid`,`inpaint` (`solid` requires `--border-color`) *Default*: `stretch`
* `--border_color`: border color to use with the `solid` border type; use BGR values from 0 to 255 *Example*: `255,0,0` is blue
* `--height`: height of crop in pixels; use with `--process_type crop` or `--process_type resize` (when used with `resize` it will distort the aspect ratio)
* `--width`: width of crop in pixels; use with `--process_type crop` or `--process_type resize` (when used with `resize` it will distort the aspect ratio)
* `--shift_y`: y (Top to bottom) amount to shift in pixels; negative values will move it up, positive will move it down; use with `--process_type crop`
* `--shift_x`: x (Left to right) amount to shift in pixels; negative values will move it left, positive will move it right; use with `--process_type crop`
* `--file_extension`: file format to output *Options*: `jpg`,`png` *Default*: `png`

### dedupe.py
Remove duplicate images from your dataset

* `--absolute`: Use absolute matching. *Default*
* `--avg_match`: average pixel difference between images (use with `--relative`) *Default*: `1.0`
* `--file_extension`: file format to output *Options*: `jpg`,`png` *Default*: `png`
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--relative`: Use relative matching.
* `--verbose`: Print progress to console.

#### Basic usage (absolute)
`python dedupe.py --input_folder path/to/input/ --output_folder path/to/output/`

#### Basic usage (relative)
`python dedupe.py --input_folder path/to/input/ --output_folder path/to/output/ --relative`

### multicrop.py
This tool produces randomized multi-scale crops. A video tutorial is [here](https://youtu.be/0yj8B2x62EA)

* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--file_extension`: file format to output *Options*: `jpg`,`png` *Default*: `png`
* `--max_size`: Maximum width and height of the crop. *Default*: `None`
* `--min_size`: Minimum width and height of the crop. *Default*: `1024`
* `--resize`: size to resize crops to (if `None` it will default to `min_size`). *Default*: `None`
* `--no_resize`: If set the crops will not be resized. *Default*: `False`
* `--verbose`: Print progress to console.

### sort.py
* `--file_extension`: file format to output *Options*: `jpg`,`png` *Default*: `png`
* `--verbose`: Print progress to console.
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--process_type`: Process to use. *Options*: `sort`,`exclude`  *Default*: `exclude`
* `--max_size`: Maximum width or height of the output images. *Default*: `2048`
* `--min_size`: Minimum width or height of the output images. *Default*: `1024`
* `--min_ratio`: Ratio of image (height/width). *Default*: `1.0`
* `--exact`: Match to exact specs. Use `--min_size` for shorter dimension, `--max_size` for longer dimension

### sort-color.py
Sorts a folder of images into separate folders based on dominant color. It will check the image's dominant color against all colors passed in, so depending on your threshold level and specified colors, you may end up with duplicate images across folders (e.g. the same sunset image in the red folder, orange folder, and yellow folder).

* `-v, --verbose`: Print progress to console.
* `-i, --input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `-o, --output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `-t, --threshold`: Threshold for color matching, lower values are more exact. *Default*: `40`
* `-c, --colors`: Comma-separated list of [W3C color names](https://en.wikipedia.org/wiki/Web_colors#X11_color_names) to check against . *Default*: `red,orange,yellow,green,blue,purple,black,white`
* `--rgb` A single color of comma-separated RGB values (e.g. 128,255,30): . *Default*: `None`

### human_filter_unique.py
Filter images by human presence and find the most unique images in a dataset. Uses YOLO11 for fast human detection with optional Moondream2 VLM fallback for edge cases. Can also filter out images containing text. Finds unique images using CLIP or DINOv2 embeddings with Farthest Point Sampling.

#### Basic Usage
```bash
# Full pipeline: detect humans, exclude text, find 100 most unique
python human_filter_unique.py \
  --input_folder ./images/ \
  --output_folder ./output/ \
  --exclude_text \
  --num_unique 100

# Keep only images WITH humans (default)
python human_filter_unique.py -i ./images/ -o ./output/ --keep humans

# Keep only images WITHOUT humans
python human_filter_unique.py -i ./images/ -o ./output/ --keep no_humans

# For M1 Macs with limited memory
python human_filter_unique.py \
  --input_folder ./images/ \
  --output_folder ./output/ \
  --human_detector yolo \
  --yolo_batch_size 4 \
  --exclude_text \
  --verbose
```

#### Key Options
* `-i, --input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `-o, --output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--verbose`: Print progress to console.
* `--mode`: Processing mode. *Options*: `full`, `human_filter`, `unique_only`. *Default*: `full`
* `--keep`: Which images to keep. *Options*: `humans`, `no_humans`. *Default*: `humans`
* `--human_detector`: Human detection method. *Options*: `yolo`, `moondream`, `hybrid`. *Default*: `hybrid`
* `--yolo_batch_size`: Batch size for YOLO (lower = less memory). *Default*: `8`
* `--exclude_text`: Enable text detection to filter out images with text.
* `--text_detector`: Text detection method. *Options*: `east`, `paddleocr`, `easyocr`, `moondream`. *Default*: `east`
* `--embedder`: Embedding model for uniqueness. *Options*: `clip`, `dinov2`. *Default*: `clip`
* `--num_unique`: Number of unique images to select. *Default*: `100`
* `--device`: Device for inference. *Options*: `auto`, `cuda`, `mps`, `cpu`. *Default*: `auto`
* `--save_discarded`: Save discarded images to a separate folder.

#### Dependencies
```bash
pip install ultralytics moondream transformers torch tqdm
# Optional for text detection:
pip install paddlepaddle paddleocr  # for PaddleOCR
pip install easyocr                  # for EasyOCR
```

### interactive.py
[YouTube Demo](https://www.youtube.com/watch?v=tUzUJNrSAu8)

### rotate.py


