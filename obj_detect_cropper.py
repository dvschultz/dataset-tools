import argparse
import numpy as np
import os
import csv
import imutils
import cv2
import random
import operator

# print(cv2.__version__)

def parse_args():
	desc = "Smarter crops using object detection (Runway YOLOv4, Colab YOLOv5)" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('--file_path', type=str,
		default='',
		help='Path to the file containing bounds data. (default: %(default)s)')

	parser.add_argument('--file_format', type=str,
		default='runway_csv',
		help='Process to use. ["runway_csv","yolo_v5"] (default: %(default)s)')

	parser.add_argument('--process_type', type=str,
		default='crop_to_square',
		help='Process to use. ["crop","crop_to_square"] (default: %(default)s)')

	parser.add_argument('--file_extension', type=str,
		default='png',
		help='file type ["png","jpg"] (default: %(default)s)')

	parser.add_argument('--min_confidence', type=float,
		default=0.5,
		help='minimum confidence score required to generate crop (default: %(default)s)')

	args = parser.parse_args()
	return args

def saveImage(img,path,filename):
	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"		
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def crop_raw(img, data):
	(h, w) = img.shape[:2]
	top = max(int( h * float(data[4]) ),0)
	bottom = int( h * float(data[5]) )
	left = max(int( h * float(data[3]) ),0)
	right = int( h * float(data[6]) )

	if args.verbose:
		print('left: {}'.format(data[3]))
		print('top: {}'.format(data[4]))
		print('right: {}'.format(data[6]))
		print('bottom: {}'.format(data[5]))
		print('left in px: {}'.format( int( h * float(data[3]) ) ) )
		print('top in px: {}'.format( int( h * float(data[4]) ) ) )
		print('right in px: {}'.format( int( h * float(data[6]) ) ) )
		print('bottom in px: {}'.format( int( h * float(data[5]) ) ) )
	
	cropped = img[top:bottom,left:right]
	return cropped

def crop_square(img, data):
	(h, w) = img.shape[:2]
	top = max(int( h * float(data[4]) ),0)
	bottom = int( h * float(data[5]) )
	left = max(int( h * float(data[3]) ),0)
	right = int( h * float(data[6]) )

	raw_w = right-left
	raw_h = bottom-top

	if(raw_w > raw_h):
		diff = (raw_w-raw_h)
		if(top-(diff/2) < 0):
			diff = diff-top
			top2 = 0
			bottom2 = bottom+diff
		elif ((diff % 2) == 0): #even
			diff = int(diff/2)
			top2 = top-diff
			bottom2 = bottom+diff
		else: #odd
			diff = int(diff/2)
			top2 = top-diff
			bottom2 = bottom+diff+1

		cropped = img[top2:bottom2,left:right]

	elif(raw_h > raw_w):
		diff = (raw_h-raw_w)
		if((left-(diff/2)) < 0):
			diff = diff-left
			left2 = 0
			right2 = right+diff
		elif ((diff % 2) == 0): #even
			diff = int(diff/2)
			left2 = left-diff
			right2 = right+diff
		else: #odd
			diff = int(diff/2)
			left2 = left-diff
			right2 = right+diff+1

		cropped = img[top:bottom,left2:right2]
	else:
		cropped = img[top:bottom,left:right]

	return cropped

def runway_csv(row):
	output_path = args.output_folder + args.process_type +"/"
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	img = cv2.imread(args.input_folder + '/' + row[0])

	if args.process_type=="crop":
		img_crop = crop_raw(img, row)
	elif args.process_type=="crop_to_square":
		img_crop = crop_square(img, row)

	return img_crop, output_path

def yolo_v5(row):
	return

def processRow(row):
	if args.file_format == "runway_csv":	
		img, output_path = runway_csv(row)
	if args.file_format == "yolo_v5":	
		yolo_v5(row)

	saveImage(img,output_path,row[0])

def main():
	global args
	global count
	global inter
	args = parse_args()
	count = int(0)
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	if args.file_format == "runway_csv":
		with open(args.file_path, newline='') as csvfile:
			csv_reader = csv.reader(csvfile)
			for row in csv_reader:
				if count != 0: #skip header
					if (args.verbose): print(', '.join(row))
					
					if float(row[2]) >= args.min_confidence:
						
						print('Processing row %d: %s' % (count, row[0]))
						processRow(row)

				count+=1


if __name__ == "__main__":
	main()