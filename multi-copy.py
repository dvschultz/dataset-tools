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

	parser.add_argument('--input_img', type=str,
		default='./input/file.png',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('--output_folder', type=str,
		default='./output/',
		help='Directory path to the output folder. (default: %(default)s)')

	parser.add_argument('--start', type=int, 
		default=1,
		help='Starting count (default: %(default)s)')

	parser.add_argument('--end', type=int, 
		default=100,
		help='Ending count (default: %(default)s)')

	# parser.add_argument('--scale', type=float, 
	# 	default=2.0,
	# 	help='Scalar value. For use with scale process type (default: %(default)s)')

	# parser.add_argument('--direction', type=str,
	# 	default='AtoB',
	# 	help='Paired Direction. For use with pix2pix process. ["AtoB","BtoA"] (default: %(default)s)')

	# parser.add_argument('--border_type', type=str,
	# 	default='stretch',
	# 	help='Border style to use when using the square process type ["stretch","reflect","solid"] (default: %(default)s)')

	# parser.add_argument('--border_color', type=str,
	# 	default='255,255,255',
	# 	help='border color to use with the `solid` border type; use bgr values (default: %(default)s)')

	# parser.add_argument('--blur_size', type=int, 
	# 	default=3,
	# 	help='Blur size. For use with "canny" process. (default: %(default)s)')

	# parser.add_argument('--mirror', action='store_true',
	# 	help='Adds mirror augmentation.')

	# parser.add_argument('--rotate', action='store_true',
	# 	help='Adds 90 degree rotation augmentation.')

	parser.add_argument('--file_extension', type=str,
		default='png',
		help='Border style to use when using the square process type ["png","jpg"] (default: %(default)s)')

	args = parser.parse_args()
	return args



def crop_square_patch(img, imgSize):
	(h, w) = img.shape[:2]

	rH = random.randint(0,h-imgSize)
	rW = random.randint(0,w-imgSize)
	cropped = img[rH:rH+imgSize,rW:rW+imgSize]

	return cropped

def makeResize(img,filename,scale):

	remakePath = args.output_folder + str(scale)+"/"
	if not os.path.exists(remakePath):
		os.makedirs(remakePath)

	img_copy = img.copy()
	img_copy = image_resize(img_copy, max = scale)

	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"
		cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])

	if (args.mirror): flipImage(img_copy,new_file,remakePath)
	if (args.rotate): rotateImage(img_copy,new_file,remakePath)


def makeSquare(img,filename,scale):
	sqPath = args.output_folder + "sq-"+str(scale)+"/"
	if not os.path.exists(sqPath):
		os.makedirs(sqPath)

	bType = cv2.BORDER_REPLICATE
	if(args.border_type == 'solid'):
		bType = cv2.BORDER_CONSTANT
	elif (args.border_type == 'reflect'):
		bType = cv2.BORDER_REFLECT
	img_sq = img.copy()
	img_sq = image_resize(img_sq, max = scale)

	bColor = [int(item) for item in args.border_color.split(',')]
	print(bColor)

	(h, w) = img_sq.shape[:2]
	if(h > w):
		# pad left/right
		diff = h-w
		if(diff%2 == 0):
			img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff/2), int(diff/2), bType,value=bColor)
		else:
			img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff/2)+1, int(diff/2), bType,value=bColor)
	elif(w > h):
		# pad top/bottom
		diff = w-h
		if(diff%2 == 0):
			img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2), 0, 0, bType,value=bColor)
		else:
			img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2)+1, 0, 0, bType,value=bColor)

	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + "-sq.png"
		cv2.imwrite(os.path.join(sqPath, new_file), img_sq, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(sqPath, new_file), img_sq, [cv2.IMWRITE_JPEG_QUALITY, 90])

	if (args.mirror): flipImage(img_sq,new_file,sqPath)
	if (args.rotate): rotateImage(img_sq,new_file,sqPath)


		

def makeSquareCropPatch(img,filename,scale):
	make_path = args.output_folder + "sq-"+str(scale)+"/"
	if not os.path.exists(make_path):
		os.makedirs(make_path)

	img_copy = img.copy()
	img_copy = crop_square_patch(img_copy,args.max_size)

	new_file = os.path.splitext(filename)[0] + ".png"
	cv2.imwrite(os.path.join(make_path, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])

	if (args.mirror): flipImage(img_copy,new_file,make_path)
	if (args.rotate): rotateImage(img_copy,new_file,make_path)


def copyImage(img,counter):
	path = args.output_folder
	if not os.path.exists(path):
		os.makedirs(path)

	imgCopy = img.copy();
	formatted = "%04d" % counter;

	if(args.file_extension == "png"):
		new_file = filename + formatted + ".png"
		cv2.imwrite(os.path.join(path, new_file), imgCopy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = filename + formatted + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), imgCopy, [cv2.IMWRITE_JPEG_QUALITY, 90])

	
def main():
	global args
	global filename
	args = parse_args()
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	filename = os.path.splitext(os.path.basename(args.input_img))[0]

	for x in range(args.start,args.end):
		img = cv2.imread(args.input_img)
		copyImage(img,x)

	# for root, subdirs, files in os.walk(args.input_folder):
	# 	print('--\nroot = ' + root)

	# 	for subdir in subdirs:
	# 		print('\t- subdirectory ' + subdir)

	# 	for filename in files:
	# 		file_path = os.path.join(root, filename)
	# 		print('\t- file %s (full path: %s)' % (filename, file_path))
			
	# 		img = cv2.imread(file_path)

	# 		if hasattr(img, 'copy'):
	# 			processImage(img,filename)
	# 			count = count + int(2)


if __name__ == "__main__":
	main()
