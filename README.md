# dataset-tools
Tools for quickly normalizing image datasets for machine learning. 

# Installation
```
git clone https://github.com/dvschultz/dataset-tools.git
cd dataset-tools
pip install -r requirements.txt
```

# Basic Usage
```
python dataset-tools.py --input_folder path/to/input/ --output_folder path/to/output/
```

# All Options
## dataset_tools.py
* `--verbose`: Print progress to console.
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--process_type`: Process to use. *Options*: `resize`,`square`,`crop`,`crop_to_square`,`canny`,`pix2pix`,`scale`,`crop_to_square_patch`,`many_squares`  *Default*: `resize`
* `--max_size`: Maximum width or height of the output images. *Default*: `512`
* `--direction`: Paired Direction. For use with pix2pix process. *Options*: `AtoB`,`BtoA`.  *Default*: `AtoB`
* `--mirror`: Adds mirror augmentation.
* `--rotate`: Adds 90 degree rotation augmentation.
* `--border_type`: Border style to use when using the `square` process type *Options*: `stretch`,`reflect`,`solid` (`solid` requires `--border-color`) *Default*: `stretch`
* `--border_color`: border color to use with the `solid` border type; use BGR values from 0 to 255 *Example*: `255,0,0` is blue
* `--height`: height of crop in pixels; use with `--process_type crop`
* `--width`: width of crop in pixels; use with `--process_type crop`
* `--shift_y`: y (Top to bottom) amount to shift in pixels; negative values will move it up, positive will move it down; use with `--process_type crop`
* `--shift_x`: x (Left to right) amount to shift in pixels; negative values will move it left, positive will move it right; use with `--process_type crop`
* `--file_extension`: file format to output *Options*: `jpg`,`png` *Default*: `png`

## sort.py
* `--file_extension`: file format to output *Options*: `jpg`,`png` *Default*: `png`
* `--verbose`: Print progress to console.
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--process_type`: Process to use. *Options*: `sort`,`exclude`  *Default*: `exclude`
* `--max_size`: Maximum width or height of the output images. *Default*: `2048`
* `--min_size`: Minimum width or height of the output images. *Default*: `1024`
* `--min_ratio`: Ratio of image (height/width). *Default*: `1.0`

## dedupe.py
Remove duplicate images from your dataset

* `--absolute`: Use absolute matching. *Default*
* `--avg_match`: average pixel difference between images (use with `--relative`) *Default*: `1.0`
* `--file_extension`: file format to output *Options*: `jpg`,`png` *Default*: `png`
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--relative`: Use relative matching.
* `--verbose`: Print progress to console.

### Basic usage (absolute)
`python dedupe.py --input_folder path/to/input/ --output_folder path/to/output/`

### Basic usage (relative)
`python dedupe.py --input_folder path/to/input/ --output_folder path/to/output/ --relative`

## rotate.py


