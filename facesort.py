import cv2
import argparse
import os

def parse_args():
    desc = "This tool sorts a collection of images based on the number of faces detected in them using opencv face detection. The result is subfolders in the output folder. It can be limited to only output between a minimum and a maximum using --min and --max. You can also sort based on eyes detected using --method eyes." 
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i','--input_folder', type=str,
        default='./input/',
        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('-o','--output_folder', type=str,
        default='./output/',
        help='Directory path to the outputs folder. (default: %(default)s)')

    parser.add_argument('--file_extension', type=str,
        default='png',
        help='file extension ["png","jpg"] (default: %(default)s)')

    parser.add_argument('--verbose', action='store_true',
        help='Print progress to console.')

    parser.add_argument('--method', type=str,
        default='faces',
        help='Method ["faces","eyes"] (default: %(default)s)')

    parser.add_argument('--min', type=int,
        default=None,
        help='Specifies a minimum number of faces to output. (default: %(default)s)')

    parser.add_argument('--max', type=int,
        default=None,
        help='Specifies a maximum number of faces to output. (default: %(default)s)')

    args = parser.parse_args()
    return args

def saveImage(img,path,filename):
    if(args.file_extension == "png"):
        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif(args.file_extension == "jpg"):
        new_file = os.path.splitext(filename)[0] + ".jpg"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def process_image(img, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectors = []
    if (args.method == 'faces'):
        detectors = [
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_alt_tree.xml',
            'haarcascade_frontalface_default.xml',
            'haarcascade_profileface.xml',
        ]
    elif (args.method == 'eyes'):
        detectors = [
            'haarcascade_eye.xml',
            'haarcascade_eye_tree_eyeglasses.xml',
        ]
    else:
        print("Unknown method: " + args.method)
        return;

    counts = []
    for detector in detectors:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + detector)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)
        counts.append(len(detected))

    detected = max(counts)
    if(args.verbose): print('\t\tdetected: ' + str(detected))

    if ((args.max == None or (args.max != None and detected <= args.max)) and (args.min == None or (args.min != None and detected >= args.min))):
        save_to = args.output_folder + '/' + str(detected)
        if(args.verbose): print('\t\tsaving to: ' + save_to)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        saveImage(img, save_to, filename)

def main():
    global args
    args = parse_args()

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
                # if(args.verbose): print('processing image: ' + filename)
                process_image(img,os.path.splitext(filename)[0])

if __name__ == "__main__":
    main()
