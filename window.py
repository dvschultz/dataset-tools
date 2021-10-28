import argparse
import numpy as np
import scipy.ndimage as pyimg
import os
import imutils
import cv2
import random

def parse_args():
	desc = "Sliding Window tool" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-ox','--offset_x', type=int,
		default=0,
		help='amount to offset on the X dimension for each slice. (default: %(default)s)')

	parser.add_argument('-oy','--offset_y', type=int,
		default=0,
		help='amount to offset on the Y dimension for each slice. (default: %(default)s)')

	parser.add_argument('-ht','--height', type=int, 
		default=None,
		help='Maximum height of the output image (for use with --process_type crop). (default: %(default)s)')

	parser.add_argument('-w','--width', type=int, 
		default=None,
		help='Maximum width of output image (for use with --process_type crop). (default: %(default)s)')

	parser.add_argument('--skip_tags', type=str, 
        default=None,
        help='comma separated color tags (for Mac only) (default: %(default)s)')

	parser.add_argument('--start_number', type=int, 
        default=0,
        help='starting index for --numbered (default: %(default)s)')

	parser.add_argument('-d','--direction', type=str,
		default='YthenX',
		help='direction to process each image ["YthenX", "XthenY"] (default: %(default)s)')

	parser.add_argument('-fe','--file_extension', type=str,
		default='png',
		help='Border style to use when using the square process type ["png","jpg"] (default: %(default)s)')

	feature_parser = parser.add_mutually_exclusive_group(required=False)
	feature_parser.add_argument('--keep_name', dest='name', action='store_true')
	feature_parser.add_argument('--numbered', dest='name', action='store_false')
	parser.set_defaults(name=True)

	args = parser.parse_args()
	return args

def saveImage(img,path,filename):
	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def main():
	global args
	global gcount
	global inter
	args = parse_args()
	gcount = args.start_number
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	if(args.skip_tags != None):
		import mac_tag

	if os.path.isdir(args.input_folder):
		print("Processing folder: " + args.input_folder)
	elif os.path.isfile(args.input_folder):
		img = cv2.imread(args.input_folder)
		filename = args.input_folder.split('/')[-1]

		if hasattr(img, 'copy'):
			if(args.verbose): print('processing image: ' + filename)  
			processImage(img,os.path.splitext(filename)[0])

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
						skipped = True
						
			
			if not skipped:
				img = cv2.imread(file_path)

				if hasattr(img, 'copy'):
					if(args.verbose): print('processing image: ' + filename)
					processImage(img,os.path.splitext(filename)[0])
			


def processImage(img,filename):
	global gcount
	output_path = args.output_folder
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	img_copy = img.copy()
	(h, w) = img.shape[:2]

	counter = 0
	y = 0
	if (args.direction == 'YthenX'):
		while(y <= (h - args.height)):
			
			x = 0
			while(x <= (w - args.width)):
				img_copy = img[y:y+args.height,x:x+args.width]
				if(args.name):
					fn = filename+'-'+str(counter)
				else:
					fn = str(gcount).zfill(9)
				saveImage(img_copy,output_path,fn)
				
				counter += 1
				gcount += 1
				x += (args.width + args.offset_x)

			y+= (args.height + args.offset_y)
			
	elif (args.direction == 'XthenY'):
		while(x <= (w - args.width)):
			
			y = 0
			while(y <= (y - args.height)):
				img_copy = img[y:y+args.height,x:x+args.width]
				if(args.name):
					fn = filename+'-'+str(counter)
				else:
					fn = str(gcount).zfill(9)
				saveImage(img_copy,output_path,fn)
				
				counter += 1
				gcount += 1
				y += (args.height + args.offset_y)

			x+= (args.width + args.offset_x)

if __name__ == "__main__":
	main()
