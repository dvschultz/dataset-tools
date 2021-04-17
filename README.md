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

## All Options
### dataset_tools.py
* `--verbose`: Print progress to console.
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--process_type`: Process to use. *Options*: `resize`,`square`,`crop`,`crop_to_square`,`canny`,`canny-pix2pix`,`scale`,`crop_square_patch`,`many_squares`  *Default*: `resize`
* `--blur_type`: Blur process to use. Use with `--process_type canny`. *Options*: `none`, `gaussian`, `median`. *Default*: `none`
* `--blur_amount`: Amount of blur to apply (use odd integers only). Use with `--blur_type`. *Default*: `1`
* `--max_size`: Maximum width or height of the output images. *Default*: `512`
* `--force_max`: forces the resize to the max size (by default `--max_size` only scales down)
* `--direction`: Paired Direction. For use with pix2pix process. *Options*: `AtoB`,`BtoA`.  *Default*: `AtoB`
* `--mirror`: Adds mirror augmentation.
* `--rotate`: Adds 90 degree rotation augmentation.
* `--border_type`: Border style to use when using the `square` process type *Options*: `stretch`,`reflect`,`solid` (`solid` requires `--border-color`) *Default*: `stretch`
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


### interactive.py
[YouTube Demo](https://www.youtube.com/watch?v=tUzUJNrSAu8)

### rotate.py


