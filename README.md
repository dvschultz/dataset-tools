# dataset-tools
Tools for quickly normalizing image datasets for machine learning. 

# Basic Usage
```
python dataset-tools.py --input_folder path/to/input/ --output_folder path/to/output/
```

# All Options
* `--verbose`: Print progress to console.
* `--input_folder`: Directory path to the inputs folder. *Default*: `./input/`
* `--output_folder`: Directory path to the outputs folder. *Default*: `./output/`
* `--process_type`: Process to use. *Options*: `resize`,`square`,`crop_to_square`,`canny`,`pix2pix`  *Default*: `resize`
* `--max_size`: Maximum width or height of the output images. *Default*: `512`
* `--direction`: Paired Direction. For use with pix2pix process. *Options*: `AtoB`,`BtoA`.  *Default*: `AtoB`
* `--mirror`: Adds mirror augmentation.
* `--rotate`: Adds 90 degree rotation augmentation.