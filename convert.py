import os
import argparse
import cv2
import numpy as np
import queue
import _thread

def saveImage(img,path,filename):
	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def processImage(img,filename):
	saveImage(img, args.output_folder,filename)

def parse_args():
    desc = "Tools to crop unnecessary space from outside of images" 
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

    parser.add_argument('-j' '--jobs', type=int,
        default=1,
        help='The number of threads to use. (default: %(default)s)')

    args = parser.parse_args()
    return args

def threadRunner(threadName):
    while(not q.empty()):
        filename = q.get()
        convertImage(threadName, filename)

def convertImage(threadName, filename):
    file_path = os.path.join(rootDir, filename)
    if(args.verbose): print('(%s) processing\t- file %s (full path: %s)' % (threadName, filename, file_path))
            
    img = cv2.imread(file_path)

    if hasattr(img, 'copy'):
        processImage(img,os.path.splitext(filename)[0])
    

def main():
    global args
    global inter
    global q
    global rootDir
    args = parse_args()
    q = queue.Queue()

    os.environ['OPENCV_IO_ENABLE_JASPER']= "true"
    inter = cv2.INTER_CUBIC

    if os.path.isdir(args.input_folder):
        print("Processing folder: " + args.input_folder)
    else:
        print("Not a working input_folder path: " + args.input_folder)
        return

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for root, subdirs, files in os.walk(args.input_folder):
        if(args.verbose): print('--\nroot = ' + root)
        rootDir = root

        for subdir in subdirs:
            if(args.verbose): print('\t- subdirectory ' + subdir)

        for filename in files:
            # add files to queue
            q.put(filename)
            
    # start threads
    for i in range(args.j__jobs):
        try:
            threadName = 'thread-' + str(i)
            if(args.verbose): print('starting thread %s' % (threadName))
            _thread.start_new_thread(threadRunner, (threadName,))
        except:
            print("error! unable to start thread.")

    while (not q.empty()):
        pass


if __name__ == "__main__":
    main()