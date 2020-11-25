import argparse
import cv2
import numpy as np
import os

def crop_image_only_outside(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def crop_dims(img,tol=0,padding=10):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return (row_start,row_end,col_start,col_end)

def saveImage(img,path,filename):
    if(args.file_extension == "png"):
        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif(args.file_extension == "jpg"):
        new_file = os.path.splitext(filename)[0] + ".jpg"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def processImage(img,filename):
    padding = args.padding

    original = img.copy()
    (h, w) = img.shape[:2]

    resized = cv2.resize(img, (int(w*args.scalar),int(h*args.scalar)), interpolation = inter)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (args.blur_size, args.blur_size), 0)

    if(args.process_type == 'canny'):
        # https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
        v = np.median(gray)
        #---- Apply automatic Canny edge detection using the computed median----
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        masked = cv2.Canny(blurred, lower, upper)

    else:
        masked = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,111,20)

    crop = crop_image_only_outside(masked)
    crop_dim = crop_dims(masked)
    crop_dim = [(int(1/args.scalar))*x for x in crop_dim]
    print(crop_dim)

    if(crop_dim[0]-padding >= 0):
        crop_dim[0]-=padding
    else:
        crop_dim[0] = 0

    if(crop_dim[1]+padding <= h):
        crop_dim[1]+=padding
    else:
        crop_dim[1] = h

    if(crop_dim[2]-padding >= 0):
        crop_dim[2]-=padding
    else:
        crop_dim[2] = 0

    if(crop_dim[3]+padding <= w):
        crop_dim[3]+=padding
    else:
        crop_dim[3] = w

    img_out = original[crop_dim[0]:crop_dim[1],crop_dim[2]:crop_dim[3]]
    saveImage(img_out,args.output_folder,filename)

    if (args.save_canny):
        saveImage(masked,args.output_folder,filename+'-canny')

def parse_args():
    desc = "Tools to crop unnecessary space from outside of images" 
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--blur_size', type=int, 
        default=5,
        help='size of blur kernel, in pixels (default: %(default)s)')

    parser.add_argument('--input_folder', type=str,
        default='./input/',
        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('--output_folder', type=str,
        default='./output/',
        help='Directory path to the outputs folder. (default: %(default)s)')

    parser.add_argument('--file_extension', type=str,
        default='png',
        help='Border style to use when using the square process type ["png","jpg"] (default: %(default)s)')

    parser.add_argument('--padding', type=int, 
        default=100,
        help='padding around crop, in pixels. (default: %(default)s)')

    parser.add_argument('--process_type', type=str,
        default='canny',
        help='Options ["canny","threshold"] (default: %(default)s)')

    parser.add_argument('--save_canny', action='store_true',
        help='Save out Canny image (for debugging)')

    parser.add_argument('--scalar', type=float, 
        default=.125,
        help='Scalar value. For use with scale process type (default: %(default)s)')

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
    padding = 100

    if os.path.isdir(args.input_folder):
        print("Processing folder: " + args.input_folder)
    else:
        print("Not a working input_folder path: " + args.input_folder)
        return;

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for root, subdirs, files in os.walk(args.input_folder):
        if(args.verbose): print('--\nroot = ' + root)

        for subdir in subdirs:
            if(args.verbose): print('\t- subdirectory ' + subdir)

        for filename in files:
            file_path = os.path.join(root, filename)
            if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))
            
            img = cv2.imread(file_path)

            if hasattr(img, 'copy'):
                if(args.verbose): print('processing image: ' + filename)  
                processImage(img,os.path.splitext(filename)[0])

if __name__ == "__main__":
    main()
