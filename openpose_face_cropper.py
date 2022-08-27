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
	desc = "Smarter crops using Openpose face detection" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--verbose', action='store_true',
		default= False,
		help='Print progress to console.')

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-b', '--bounds_file_path', type=str,
		default='',
		help='Path to the file containing bounds data. (default: %(default)s)')

	# parser.add_argument('--process_type', type=str,
	# 	default='crop_to_square',
	# 	help='Process to use. ["crop","crop_to_square"] (default: %(default)s)')

	parser.add_argument('--file_extension', type=str,
		default='png',
		help='file type ["png","jpg"] (default: %(default)s)')

	args = parser.parse_args()
	return args

def saveImage(img,path,filename):
	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"		
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def get_bounding_box(x,y):
	return (int(min(x)),int(min(y)),int(max(x)),int(max(y)))


def crop_raw(img, data):
	(h, w) = img.shape[:2]
	top = max(int( h * float(data[4]) ),0)
	bottom = min(int( h * float(data[5]) ),h)
	left = max(int( h * float(data[3]) ),0)
	right = min(int( h * float(data[6]) ),w)

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

def crop_square(img, bb, pad):
	(h, w) = img.shape[:2]
	bb_h = bb[3]-bb[1]
	bb_w = bb[2]-bb[0]
	
	#account for forehead
	bb_h = bb_h *1.33

	#create new max y
	bb = list(bb)
	bb[1] = int(bb[3]-bb_h)
	bb[1] = max(0, bb[1])
	bb = tuple(bb)
	
	#get centerpoint of bounding box
	bb_c = (int((bb[2]+bb[0])/2), int((bb[3]+bb[1])/2) )

	if(bb_h > bb_w):
		new_h = int(bb_h + (bb_h * pad))
		min_y = max(0, int(-(new_h/2)))
		max_y = min(h, int(bb_c[1]+(new_h/2)))
		min_x = max(0, int(bb_c[0]-(new_h/2)))
		max_x = min(w, int(bb_c[0]+(new_h/2)))

		new_h2 = min(min(max_y-bb_c[1], bb_c[1]-min_y), min(max_x-bb_c[0], bb_c[0]-min_x))

		cropped = img[int(bb_c[1]-(new_h2)):int(bb_c[1]+(new_h2)),int(bb_c[0]-(new_h2)):int(bb_c[0]+(new_h2))]

	elif(bb_w > bb_h):
		new_w = int(bb_w + (bb_w * pad))
		min_y = max(0, int(bb_c[0]-(new_w/2)))
		max_y = min(h, int(bb_c[0]+(new_w/2)))
		min_x = max(0, int(bb_c[1]-(new_w/2)))
		max_x = min(w, int(bb_c[1]+(new_w/2)))

		new_w2 = min(min(max_y-bb_c[1], bb_c[1]-min_y), min(max_x-bb_c[0], bb_c[0]-min_x))

		cropped = img[int(bb_c[0]-(new_w2)):int(bb_c[0]+(new_w2)),int(bb_c[1]-(new_w2)):int(bb_c[1]+(new_w2))]
	else:
		new_h = int(bb_h + (bb_h * pad))
		min_y = max(0, int(-(new_h/2)))
		max_y = min(h, int(bb_c[1]+(new_h/2)))
		min_x = max(0, int(bb_c[0]-(new_h/2)))
		max_x = min(w, int(bb_c[0]+(new_h/2)))

		new_h2 = min(min(max_y-bb_c[1], bb_c[1]-min_y), min(max_x-bb_c[0], bb_c[0]-min_x))
		cropped = img[int(bb_c[1]-(new_h2)):int(bb_c[1]+(new_h2)),int(bb_c[0]-(new_h2)):int(bb_c[0]+(new_h2))]

	return cropped

def process(path,data):
	print(path)
	# print(data)
	img = cv2.imread(path)

	face_keypoints = data

	x = []
	y = []
	c = []
	count = 1
	for i, v in enumerate(face_keypoints):
		if (count == 1):
			x.append(v)
		elif(count == 2):
			y.append(v)
		else:
			c.append(v)

		count += 1
		if(count > 3):
			count = 1

	# print(x)
	# print(y)

	if( (sum(x) > 0) and (sum(y) > 0) ):
		bb = get_bounding_box(x,y)
		cropped = crop_square(img,bb,0.5)
		# print(bb)

		if (cropped is not None) and (len(cropped) > 0):
			saveImage(cropped, args.outpath, path.split('/')[-1])


def main():
	global args
	global count
	global inter
	args = parse_args()
	count = int(0)
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	data = np.load(args.bounds_file_path,allow_pickle=True).tolist()

	args.outpath = os.path.join(args.output_folder,'1s-frames-facecrops')
	if not os.path.exists(args.outpath):
		os.makedirs(args.outpath)

	# print()

	for i in range(len(data)):
		path = os.path.join(args.input_folder, data[i][0].replace('_keypoints.json', '.png'))
		if os.path.exists(path):

			# face_keypoints = data[i][1][0]['face_keypoints_2d']
			face_keypoints_type = type(data[i][1][0])
			
			if(face_keypoints_type is dict):
				process(path,data[i][1][0]['face_keypoints_2d'])
			else:
				print('no keypoint data for: ', path)


if __name__ == "__main__":
	main()