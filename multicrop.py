import os
import argparse
import cv2
import numpy as np
import shutil

def saveImage(img,path,filename):
	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

def processImage(img,filename):
    fn = filename.replace(' ','-').replace('&','_').replace('.','')
    (h, w) = img.shape[:2]
    min_size = args.min_size

    if(args.verbose): print('max_size: {}'.format(str(max_size)))
    if(args.max_size):
        max_size = args.max_size
    else:
        max_size = min(h,w)

    if (min_size > max_size):
        print('image is too small for set min_size: ' + fn);
        return

    print('min_size: {}'.format(str(min_size)))
    print('max_size: {}'.format(str(max_size)))

    if (min_size != max_size):
        original = img.copy()
        # generate random patch size between min ad max
        r = np.random.randint(min_size, max_size, size=args.how_many)
        
        for i in range(args.how_many):
            
            start = (np.random.randint(0,h-r[i]),np.random.randint(0,w-r[i]))
            patch = img[start[0]:start[0]+r[i],start[1]:start[1]+r[i]]

            if not args.no_resize:
                if(args.resize):
                    patch = cv2.resize(patch, (args.resize,args.resize), interpolation = inter)
                else:
                    patch = cv2.resize(patch, (min_size,min_size), interpolation = inter)

            saveImage(patch, args.output_folder,fn+'-'+str(i))
    else:
        print('image is exact size: ' + fn)
        saveImage(img, args.output_folder,fn)

def parse_args():
    desc = "Tools to crop random patches for images" 
    parser = argparse.ArgumentParser(description="Tools to crop random patches for images")

    parser.add_argument('-f','--file_extension', type=str,
        default='png',
        help='Border style to use when using the square process type ["png","jpg"] (default: %(default)s)')

    parser.add_argument('--how_many', type=int, 
        default=2,
        help='How many random crops to create (default: %(default)s)')

    parser.add_argument('-i','--input_folder', type=str,
        default='./input/',
        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('--min_size', type=int, 
        default=1024,
        help='Minimum width or height of the cropped images. (default: %(default)s)')

    parser.add_argument('--max_size', type=int, 
        default=None,
        help='Maximum width or height of the cropped images. (default: %(default)s)')

    parser.add_argument('--no_resize', action='store_true',
        help='Do not resize patches.')

    parser.add_argument('-o','--output_folder', type=str,
        default='./output/',
        help='Directory path to the outputs folder. (default: %(default)s)')

    parser.add_argument('--resize', type=int, 
        default=None,
        help='Minimum width or height of the cropped images. (default: %(default)s)')
    
    parser.add_argument('--skip_tags', type=str, 
        default=None,
        help='comma separated color tags (for Mac only) (default: %(default)s)')

    parser.add_argument('--verbose', action='store_true',
        help='Print progress to console.')

    args = parser.parse_args()
    return args

def main():
    global args
    global inter
    args = parse_args()

    os.environ['OPENCV_IO_ENABLE_JASPER']= "true"
    inter = cv2.INTER_CUBIC

    if os.path.isdir(args.input_folder):
        print("Processing folder: " + args.input_folder)
    else:
        print("Not a working input_folder path: " + args.input_folder)
        return

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if(args.skip_tags != None):
        import mac_tag

    for root, subdirs, files in os.walk(args.input_folder):
        if(args.verbose): print('--\nroot = ' + root)

        for subdir in subdirs:
            if(args.verbose): print('\t- subdirectory ' + subdir)

        for filename in files:
            skipped = False
            file_path = os.path.join(root, filename)
            if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))

            if(args.skip_tags != None):
                tags = [str(item) for item in args.skip_tags.split(',')]
                # tags = mac_tag.get(file_path)
                # print(tags)
                for tag in tags:
                    matches = mac_tag.match(tag,file_path)
                    if(file_path in matches):
                        print('skipping file: ' + filename)
                        # new_path = os.path.join(args.output_folder, filename)
                        # shutil.copy2(file_path,new_path)
                        # mac_tag.add([tag],[new_path])
                        skipped = True
                        continue
            
            if not skipped:
                img = cv2.imread(file_path)

                if hasattr(img, 'copy'):
                    if(args.verbose): print('processing image: ' + filename)  
                    processImage(img,os.path.splitext(filename)[0])

if __name__ == "__main__":
    main()