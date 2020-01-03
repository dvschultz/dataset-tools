import argparse
import numpy as np
import os
import imutils
import cv2
import random

# print(cv2.__version__)

def parse_args():
	desc = "Tools to normalize an image dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('--process_type', type=str,
		default='rotate',
		help='Process to use. ["rotate","resize","scale"] (default: %(default)s)')

	parser.add_argument('--max_size', type=int, 
		default=512,
		help='Maximum width or height of the output images. (default: %(default)s)')

	parser.add_argument('--scale', type=float, 
		default=2.0,
		help='Scalar value. For use with scale process type (default: %(default)s)')
	
	parser.add_argument('--mirror', action='store_true',
		help='Adds mirror augmentation.')

	parser.add_argument('--file_extension', type=str,
		default='png',
		help='Border style to use when using the square process type ["png","jpg"] (default: %(default)s)')

	args = parser.parse_args()
	return args


def image_resize(image, width = None, height = None, max = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if max is not None:
    	if w > h:
    		# produce
    		r = max / float(w)
    		dim = (max, int(h * r))
    	elif h > w:
    		r = max / float(h)
    		dim = (int(w * r), max)
    	else :
    		dim = (max, max)

    else: 
	    # if both the width and height are None, then return the
	    # original image
	    if width is None and height is None:
	        return image

	    # check to see if the width is None
	    if width is None:
	        # calculate the ratio of the height and construct the
	        # dimensions
	        r = height / float(h)
	        dim = (int(w * r), height)

	    # otherwise, the height is None
	    else:
	        # calculate the ratio of the width and construct the
	        # dimensions
	        r = width / float(w)
	        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def image_scale(image, scalar = 1.0):
	(h, w) = image.shape[:2]
	dim = (int(w*scalar),int(h*scalar))
	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)
	
	# return the resized image
	return resized

def makeResize(img,filename,scale):

	remakePath = args.output_folder + str(scale)+"/"
	if not os.path.exists(remakePath):
		os.makedirs(remakePath)

	img_copy = img.copy()
	img_copy = image_resize(img_copy, max = scale)

	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"
		# cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
		rotateImage(img_copy,new_file,remakePath)
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		# cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])
		rotateImage(img_copy,new_file,remakePath)

	
	# if (args.mirror): flipImage(img_copy,new_file,remakePath)

def makeRotate(img,filename):
	remakePath = args.output_folder +"rotate/"
	if not os.path.exists(remakePath):
		os.makedirs(remakePath)

	new_file = os.path.splitext(filename)[0] + ".png"
	rotateImage(img,new_file,remakePath)

def makeScale(img,filename,scale):

	remakePath = args.output_folder + "scale_"+str(scale)+"/"
	if not os.path.exists(remakePath):
		os.makedirs(remakePath)

	img_copy = img.copy()
	
	img_copy = image_scale(img_copy, scale)

	new_file = os.path.splitext(filename)[0] + ".png"
	# cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])

	# if (args.mirror): flipImage(img_copy,new_file,remakePath)
	rotateImage(img_copy,new_file,remakePath)


def flipImage(img,filename,path):
	flip_img = cv2.flip(img, 1)
	flip_file = os.path.splitext(filename)[0] + "-flipped.png"
	cv2.imwrite(os.path.join(path, flip_file), flip_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def rotateImage(img,filename,path):
	r = img.copy() 
	r = imutils.rotate_bound(r, 90)
	r_file = os.path.splitext(filename)[0] + "-rot90.png"
	cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])

	# r = imutils.rotate_bound(r, 90)
	# r_file = os.path.splitext(filename)[0] + "-rot180.png"
	# cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])

	# r = imutils.rotate_bound(r, 90)
	# r_file = os.path.splitext(filename)[0] + "-rot270.png"
	# cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def processImage(img,filename):

	if args.process_type == "rotate":	
		makeRotate(img,filename)
	if args.process_type == "resize":	
		makeResize(img,filename,args.max_size)
	if args.process_type == "scale":
		makeScale(img,filename,args.scale)

def main():
	global args
	global count
	global inter
	args = parse_args()
	count = int(0)
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	for root, subdirs, files in os.walk(args.input_folder):
		print('--\nroot = ' + root)

		for subdir in subdirs:
			print('\t- subdirectory ' + subdir)

		for filename in files:
			file_path = os.path.join(root, filename)
			print('\t- file %s (full path: %s)' % (filename, file_path))
			
			img = cv2.imread(file_path)

			if hasattr(img, 'copy'):
				processImage(img,filename)
				count = count + int(2)


if __name__ == "__main__":
	main()
