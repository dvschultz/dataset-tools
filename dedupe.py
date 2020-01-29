import argparse
import numpy as np
import os
import imutils
import cv2
import random

# print(cv2.__version__)

def parse_args():
	desc = "Dedupe imageset" 
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
		default='exclude',
		help='Process to use. ["exclude","sort"] (default: %(default)s)')


	parser.add_argument('--file_extension', type=str,
		default='png',
		help='file type ["png","jpg"] (default: %(default)s)')



	args = parser.parse_args()
	return args

def compare(img1,img2):
	if(img1.shape != img2.shape):
		return False
	else:
		difference = cv2.subtract(img1, img2)    
		return not np.any(difference)

def exclude(imgs,filenames):
	path = args.output_folder + "exclude/"
	if not os.path.exists(path):
		os.makedirs(path)

	i = 0
	while i < len(imgs):
		img = imgs[i]
		filename = filenames[i]

		for i2 in range(i+1,len(imgs)):
			try: 
				img2 = imgs[i2]
				if compare(img,img2):
					print (filenames[i] + " matches " + filenames[i2])
					del imgs[i2]
					del filenames[i2]
			except IndexError:
				print("") 
			
		if(args.file_extension == "png"):
			new_file = os.path.splitext(filename)[0] + ".png"
			cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
		else:
			new_file = os.path.splitext(filename)[0] + ".jpg"
			cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

		i += 1




def sort(imgs):
	#TODO
	print("skip")
	# make_path1 = args.output_folder + "yes/"
	# make_path2 = args.output_folder + "no/"
	# if not os.path.exists(make_path1):
	# 	os.makedirs(make_path1)
	# if not os.path.exists(make_path2):
	# 	os.makedirs(make_path2)

	# (h, w) = img.shape[:2]
	# ratio = h/w

	# if(args.exact == True):
	# 	if((ratio >= 1.0) and (h == args.max_size) and (w == args.min_size)):
	# 		path = make_path1
	# 	elif((ratio < 1.0) and (w == args.max_size) and (h == args.min_size)):
	# 		path = make_path1
	# 	else:
	# 		path = make_path2
	# else:
	# 	#only works with ratio right now
	# 	if(ratio>=args.min_ratio):
	# 		path = make_path1
	# 	else:
	# 		path = make_path2

	# if(args.file_extension == "png"):
	# 	new_file = os.path.splitext(filename)[0] + ".png"
	# 	cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	# else:
	# 	new_file = os.path.splitext(filename)[0] + ".jpg"
	# 	cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def processImage(imgs,filenames):
	if args.process_type == "exclude":	
		exclude(imgs,filenames)
	if args.process_type == "sort":	
		sort(imgs,filenames)

def main():
	global args
	global count
	global inter
	args = parse_args()
	count = int(0)
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	imgs = []
	filenames = []
	for root, subdirs, files in os.walk(args.input_folder):
		print('--\nroot = ' + root)

		for subdir in subdirs:
			print('\t- subdirectory ' + subdir)

		for filename in files:
			file_path = os.path.join(root, filename)
			print('\t- file %s (full path: %s)' % (filename, file_path))
			
			imgs.append(cv2.imread(file_path))
			filenames.append(filename);

	processImage(imgs,filenames)


if __name__ == "__main__":
	main()