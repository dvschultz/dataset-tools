# Generated Docs 📜
⚠️ Do not modify this file because it will be overwritten automatically
## convert.py
```bash
usage: convert.py [-h] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER]
                  [--file_extension FILE_EXTENSION] [--verbose]

Tools to crop unnecessary space from outside of images

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  --file_extension FILE_EXTENSION
                        file extension ["png","jpg"] (default: png)
  --verbose             Print progress to console.
```
## list-remove.py
```bash
usage: list-remove.py [-h] [--verbose] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER]
                      [-of ORDERED_FILE] [--file_extension FILE_EXTENSION]

Tools to normalize an image dataset

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Print progress to console.
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  -of ORDERED_FILE, --ordered_file ORDERED_FILE
                        Process to use. ["rotate","resize","scale"] (default:
                        rotate)
  --file_extension FILE_EXTENSION
                        file extension ["png","jpg"] (default: png)
```
## obj_detect_cropper.py
```bash
usage: obj_detect_cropper.py [-h] [--verbose] [--input_folder INPUT_FOLDER]
                             [--output_folder OUTPUT_FOLDER]
                             [--bounds_file_path BOUNDS_FILE_PATH]
                             [--file_format FILE_FORMAT]
                             [--process_type PROCESS_TYPE]
                             [--file_extension FILE_EXTENSION]
                             [--min_confidence MIN_CONFIDENCE]

Smarter crops using object detection (Runway YOLOv4, Colab YOLOv5)

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Print progress to console.
  --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  --bounds_file_path BOUNDS_FILE_PATH
                        Path to the file containing bounds data. (default: )
  --file_format FILE_FORMAT
                        Process to use. ["runway_csv","yolo_v5"] (default:
                        runway_csv)
  --process_type PROCESS_TYPE
                        Process to use. ["crop","crop_to_square"] (default:
                        crop_to_square)
  --file_extension FILE_EXTENSION
                        file type ["png","jpg"] (default: png)
  --min_confidence MIN_CONFIDENCE
                        minimum confidence score required to generate crop
                        (default: 0.5)
```
## multi-copy.py
```bash
usage: multi-copy.py [-h] [--verbose] [--input_img INPUT_IMG]
                     [--output_folder OUTPUT_FOLDER] [--start START]
                     [--end END] [--file_extension FILE_EXTENSION]

Tools to normalize an image dataset

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Print progress to console.
  --input_img INPUT_IMG
                        Directory path to the inputs folder. (default:
                        ./input/file.png)
  --output_folder OUTPUT_FOLDER
                        Directory path to the output folder. (default:
                        ./output/)
  --start START         Starting count (default: 1)
  --end END             Ending count (default: 100)
  --file_extension FILE_EXTENSION
                        Border style to use when using the square process type
                        ["png","jpg"] (default: png)
```
## rotate.py
```bash
usage: rotate.py [-h] [--verbose] [--input_folder INPUT_FOLDER]
                 [--output_folder OUTPUT_FOLDER] [--process_type PROCESS_TYPE]
                 [--max_size MAX_SIZE] [--scale SCALE] [--mirror]
                 [--file_extension FILE_EXTENSION]

Tools to normalize an image dataset

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Print progress to console.
  --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  --process_type PROCESS_TYPE
                        Process to use. ["rotate","resize","scale"] (default:
                        rotate)
  --max_size MAX_SIZE   Maximum width or height of the output images.
                        (default: 512)
  --scale SCALE         Scalar value. For use with scale process type
                        (default: 2.0)
  --mirror              Adds mirror augmentation.
  --file_extension FILE_EXTENSION
                        Border style to use when using the square process type
                        ["png","jpg"] (default: png)
```
## psd.py
```bash
usage: psd.py [-h] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [-v]

Tools to normalize an image dataset

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  -v, --verbose         Print progress to console.
```
## multicrop.py
```bash
usage: multicrop.py [-h] [-f FILE_EXTENSION] [--how_many HOW_MANY]
                    [-i INPUT_FOLDER] [--min_size MIN_SIZE]
                    [--max_size MAX_SIZE] [--no_resize] [-o OUTPUT_FOLDER]
                    [--resize RESIZE] [--verbose]

Tools to crop random patches for images

optional arguments:
  -h, --help            show this help message and exit
  -f FILE_EXTENSION, --file_extension FILE_EXTENSION
                        Border style to use when using the square process type
                        ["png","jpg"] (default: png)
  --how_many HOW_MANY   How many random crops to create (default: 2)
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  --min_size MIN_SIZE   Minimum width or height of the cropped images.
                        (default: 1024)
  --max_size MAX_SIZE   Maximum width or height of the cropped images.
                        (default: None)
  --no_resize           Do not resize patches.
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  --resize RESIZE       Minimum width or height of the cropped images.
                        (default: None)
  --verbose             Print progress to console.
```
## dedupe.py
```bash
usage: dedupe.py [-h] [--verbose] [--input_folder INPUT_FOLDER]
                 [--output_folder OUTPUT_FOLDER] [--process_type PROCESS_TYPE]
                 [--file_extension FILE_EXTENSION] [--avg_match AVG_MATCH]
                 [--absolute | --relative]

Dedupe imageset

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Print progress to console.
  --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  --process_type PROCESS_TYPE
                        Process to use. ["exclude"] (default: exclude)
  --file_extension FILE_EXTENSION
                        file type ["png","jpg"] (default: png)
  --avg_match AVG_MATCH
                        average pixel difference between images (use with
                        --relative) (default: 1.0)
  --absolute
  --relative
```
## crop_bounds.py
```bash
usage: crop_bounds.py [-h] [--blur_size BLUR_SIZE] [--dilate_iter DILATE_ITER]
                      [--erode_iter ERODE_ITER] [-i INPUT_FOLDER]
                      [--keep_original] [--max_angle MAX_ANGLE]
                      [--min_height MIN_HEIGHT] [--min_width MIN_WIDTH]
                      [--min_size MIN_SIZE] [-o OUTPUT_FOLDER]
                      [-f FILE_EXTENSION] [--fill_boxes] [--min MIN]
                      [--padding PADDING] [--precrop PRECROP]
                      [-p PROCESS_TYPE] [--remove_text]
                      [--replace_white REPLACE_WHITE] [--resize RESIZE]
                      [--rotate] [--img_debug] [--scalar SCALAR]
                      [--skip_tags SKIP_TAGS] [--text_ar TEXT_AR]
                      [--text_color TEXT_COLOR] [--thresh_block THRESH_BLOCK]
                      [--thresh_c THRESH_C] [--verbose]

Tools to crop unnecessary space from outside of images

optional arguments:
  -h, --help            show this help message and exit
  --blur_size BLUR_SIZE
                        size of blur kernel, in pixels (default: 3)
  --dilate_iter DILATE_ITER
                        iterations for dilation kernel (increasing can help
                        with tracked type) (default: 1)
  --erode_iter ERODE_ITER
                        iterations for erode kernel (increasing can help with
                        tracked type) (default: 1)
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  --keep_original       Save out original image alongside crops (for
                        comparison or debugging)
  --max_angle MAX_ANGLE
                        Maximum rotation to output. For use with --rotate
                        (default: None)
  --min_height MIN_HEIGHT
                        minimum height contour, in pixels (default: None)
  --min_width MIN_WIDTH
                        minimum width contour, in pixels (default: None)
  --min_size MIN_SIZE   minimum width contour, in pixels (default: 1024)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  -f FILE_EXTENSION, --file_extension FILE_EXTENSION
                        Border style to use when using the square process type
                        ["png","jpg"] (default: png)
  --fill_boxes          Fill box diagrams when using --rotate (for comparison
                        or debugging)
  --min MIN             min pixel color (default: 127)
  --padding PADDING     padding around crop, in pixels. (default: 100)
  --precrop PRECROP     crop image before processing (in pixels).
                        Top,Bottom,Left,Right; example: "10,20,10,10"
                        (default: None)
  -p PROCESS_TYPE, --process_type PROCESS_TYPE
                        Options ["canny","threshold","contours"] (default:
                        contours)
  --remove_text         Remove text from image
  --replace_white REPLACE_WHITE
                        color to replace text blocks with; use bgr values
                        (default: None)
  --resize RESIZE       resize longest side, in pixels (default: None)
  --rotate              Save out original image alongside crops (for
                        comparison or debugging)
  --img_debug           Save out masked image (for debugging)
  --scalar SCALAR       Scalar value. For use with scale process type
                        (default: 0.125)
  --skip_tags SKIP_TAGS
                        comma separated color tags (for Mac only) (default:
                        None)
  --text_ar TEXT_AR     aspect ratio for text detection (reduce to find
                        smaller bits of text) (default: 3)
  --text_color TEXT_COLOR
                        options: black, brown (default: black)
  --thresh_block THRESH_BLOCK
                        block size for thresholding (default: 11)
  --thresh_c THRESH_C   c value for thresholding (default: 2)
  --verbose             Print progress to console.
```
## sort.py
```bash
usage: sort.py [-h] [--verbose] [--exact] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER]
               [-p PROCESS_TYPE] [--max_size MAX_SIZE] [--max_dist MAX_DIST]
               [--min_size MIN_SIZE] [--min_ratio MIN_RATIO]
               [-f FILE_EXTENSION] [--start_img START_IMG] [--use_gpu]

Tools to normalize an image dataset

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Print progress to console.
  --exact               match to exact specs
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  -p PROCESS_TYPE, --process_type PROCESS_TYPE
                        Process to use. ["exclude","sort","tagsort","lpips"]
                        (default: exclude)
  --max_size MAX_SIZE   Maximum width or height of the output images.
                        (default: 2048)
  --max_dist MAX_DIST   Maximum distance between two images (for lpips
                        process). (default: 1.0)
  --min_size MIN_SIZE   Maximum width or height of the output images.
                        (default: 1024)
  --min_ratio MIN_RATIO
                        Ratio of image (height/width). (default: 1.0)
  -f FILE_EXTENSION, --file_extension FILE_EXTENSION
                        file type ["png","jpg"] (default: png)
  --start_img START_IMG
                        image for comparison (for lpips process)
  --use_gpu             use GPU (for lpips process)
```
## dataset-tools.py
```bash
usage: dataset-tools.py [-h] [--verbose] [--force_max] [-i INPUT_FOLDER]
                        [-o OUTPUT_FOLDER] [-p PROCESS_TYPE]
                        [--blur_type BLUR_TYPE] [--blur_amount BLUR_AMOUNT]
                        [--max_size MAX_SIZE] [--height HEIGHT]
                        [--width WIDTH] [--shift_y SHIFT_Y]
                        [--v_align V_ALIGN] [--h_align H_ALIGN]
                        [--shift_x SHIFT_X] [--scale SCALE]
                        [--skip_tags SKIP_TAGS] [--direction DIRECTION]
                        [--border_type BORDER_TYPE]
                        [--border_color BORDER_COLOR] [--mirror] [--rotate]
                        [--file_extension FILE_EXTENSION]
                        [--keep_name | --numbered]

Tools to normalize an image dataset

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Print progress to console.
  --force_max           Force max size
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Directory path to the inputs folder. (default:
                        ./input/)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Directory path to the outputs folder. (default:
                        ./output/)
  -p PROCESS_TYPE, --process_type PROCESS_TYPE
                        Process to use.
                        ["resize","square","crop_to_square","canny","canny-pix
                        2pix","crop_square_patch","scale","many_squares","crop
                        ","distance"] (default: resize)
  --blur_type BLUR_TYPE
                        Blur process to use. Use with --process_type canny.
                        ["none","gaussian","median"] (default: none)
  --blur_amount BLUR_AMOUNT
                        Amount of blur to apply (use odd numbers). Use with
                        --blur_type. (default: 1)
  --max_size MAX_SIZE   Maximum width or height of the output images.
                        (default: 1024)
  --height HEIGHT       Maximum height of the output image (for use with
                        --process_type crop). (default: None)
  --width WIDTH         Maximum width of output image (for use with
                        --process_type crop). (default: None)
  --shift_y SHIFT_Y     y coordinate shift (for use with --process_type crop).
                        (default: 0)
  --v_align V_ALIGN     -vertical alignment options: top, bottom, center (for
                        use with --process_type crop_to_square). (default:
                        center)
  --h_align H_ALIGN     -vertical alignment options: left, right, center (for
                        use with --process_type crop_to_square). (default:
                        center)
  --shift_x SHIFT_X     x coordinate shift (for use with --process_type crop).
                        (default: 0)
  --scale SCALE         Scalar value. For use with scale process type
                        (default: 2.0)
  --skip_tags SKIP_TAGS
                        comma separated color tags (for Mac only) (default:
                        None)
  --direction DIRECTION
                        Paired Direction. For use with pix2pix process.
                        ["AtoB","BtoA"] (default: AtoB)
  --border_type BORDER_TYPE
                        Border style to use when using the square process type
                        ["stretch","reflect","solid"] (default: stretch)
  --border_color BORDER_COLOR
                        border color to use with the `solid` border type; use
                        bgr values (default: 255,255,255)
  --mirror              Adds mirror augmentation.
  --rotate              Adds 90 degree rotation augmentation.
  --file_extension FILE_EXTENSION
                        Border style to use when using the square process type
                        ["png","jpg"] (default: png)
  --keep_name
  --numbered
```
