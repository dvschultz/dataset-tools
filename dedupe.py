import argparse
import numpy as np
import os
import imutils
import cv2
import random
import operator

# print(cv2.__version__)
from utils.load_images import load_images_multi_thread


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
		help='Process to use. ["exclude"] (default: %(default)s)')

	parser.add_argument('--file_extension', type=str,
		default='png',
		help='file type ["png","jpg"] (default: %(default)s)')

	parser.add_argument('--avg_match', type=float,
		default=1.0,
		help='average pixel difference between images (use with --relative) (default: %(default)s)')

	parser.add_argument('-j' '--jobs', type=int,
		default=1,
		help='The number of threads to use. (default: %(default)s)')

	feature_parser = parser.add_mutually_exclusive_group(required=False)
	feature_parser.add_argument('--absolute', dest='absolute', action='store_true')
	feature_parser.add_argument('--relative', dest='absolute', action='store_false')
	parser.set_defaults(absolute=True)

	args = parser.parse_args()
	return args

def compare(img1,img2):
	test = False
	difference = cv2.absdiff(img1, img2)
	if(args.absolute):	
		return not np.any(difference)
	else:
		return np.divide(np.sum(difference),img1.shape[0]*img1.shape[1]) <= args.avg_match

		#way too greedy
		#return np.allclose(img1,img2,2,2)


def exclude(imgs,filenames):
	path = args.output_folder + "exclude/"
	if not os.path.exists(path):
		os.makedirs(path)

	i = 0
	print("avg_match" + str(args.avg_match))
	print("processing...")
	print("total images: " + str(len(imgs)))

	while i < len(imgs):
		img = imgs[i][0]
		filename = imgs[i][1]
		(h1, w1) = img.shape[:2]

		print("matching to: " + filename)
		print( str(i) + "/" + str(len(imgs)) )

		i2 = i+1
		while i2 < len(imgs):
			popped = False
			img2 = imgs[i2][0]
			filename2 = imgs[i2][1]
			(h2, w2) = img2.shape[:2]

			# print ('comparing '+filename + " to " + filename2)
			if (h1 == h2) and (w1 == w2):
				if compare(img,img2):
					print (filename + " matches " + filename2)
					popped = True
					imgs.pop(i2)

			if not popped:
				i2 += 1

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
	print("loading images...")
	to_load = []
	for root, subdirs, files in os.walk(args.input_folder):
		if(args.verbose): print('--\nroot = ' + root)

		for subdir in subdirs:
			if(args.verbose): print('\t- subdirectory ' + subdir)

		for filename in files:
			file_path = os.path.join(root, filename)
			to_load.append(file_path)
			filenames.append(filename)

	loaded_images = load_images_multi_thread(to_load, args.j__jobs, args.verbose)
	assert len(loaded_images) == len(to_load) == len(filenames)
	for i in range(len(loaded_images)):
		imgs.append([loaded_images[i], filenames[i]])

	print("sorting images...")
	imgs.sort(key=operator.itemgetter(1))	
	# for n in range(4):
	# 	print(imgs[n][1])		
	processImage(imgs,filenames)


if __name__ == "__main__":
	main()