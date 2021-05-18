import argparse
import numpy as np
import os
import imutils
import cv2
import random
import shutil

def parse_args():
	desc = "Tools to normalize an image dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('-v','--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('--exact', action='store_true',
		help='match to exact specs')

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-p','--process_type', type=str,
		default='exclude',
		help='Process to use. ["exclude","sort","tagsort","lpips","channels"] (default: %(default)s)')

	parser.add_argument('--max_size', type=int, 
		default=2048,
		help='Maximum width or height of the output images. (default: %(default)s)')

	parser.add_argument('--max_dist', type=float, 
		default=1.0,
		help='Maximum distance between two images (for lpips process). (default: %(default)s)')

	parser.add_argument('--min_size', type=int, 
		default=1024,
		help='Maximum width or height of the output images. (default: %(default)s)')

	parser.add_argument('--min_ratio', type=float, 
		default=1.0,
		help='Ratio of image (height/width). (default: %(default)s)')

	parser.add_argument('-n','--network', type=str,
		default='alex',
		help='Network to use for the LPIPS sort process. Options: alex, vgg, squeeze (default: %(default)s)')

	parser.add_argument('-f','--file_extension', type=str,
		default='png',
		help='file type ["png","jpg"] (default: %(default)s)')

	parser.add_argument('--skip_tags', type=str, 
        default=None,
        help='comma separated color tags (for Mac only) (default: %(default)s)')

	parser.add_argument('--start_img', type=str,
		help='image for comparison (for lpips process)')

	parser.add_argument('--use_gpu', action='store_true', 
		help='use GPU (for lpips process)')

	args = parser.parse_args()
	return args

def saveImage(img,path,filename):
    if(args.file_extension == "png"):
        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif(args.file_extension == "jpg"):
        new_file = os.path.splitext(filename)[0] + ".jpg"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def exclude(img,filename):
	make_path = args.output_folder + "exclude_"+str(args.min_size)+"-"+str(args.max_size)+"/"
	if not os.path.exists(make_path):
		os.makedirs(make_path)

	(h, w) = img.shape[:2]

	if((h >= args.min_size) and (h <= args.max_size) and (w >= args.min_size) and (w <= args.max_size)):

		if(args.file_extension == "png"):
			new_file = os.path.splitext(filename)[0] + ".png"
			cv2.imwrite(os.path.join(make_path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
		else:
			new_file = os.path.splitext(filename)[0] + ".jpg"
			cv2.imwrite(os.path.join(make_path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def gray_color(img,filename):
	gray_path = args.output_folder + "gray/"
	color_path = args.output_folder + "color/"
	if not os.path.exists(gray_path):
		os.makedirs(gray_path)
	if not os.path.exists(color_path):
		os.makedirs(color_path)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mean, std = cv2.meanStdDev(hsv)
	if(args.verbose): print(mean[1],std[1])

	if(mean[1] >= 44.0):
		saveImage(img,color_path,filename)
	elif(mean[1] <= 10.0): 
		saveImage(img,gray_path,filename)
	elif(std[1] >= 30.0):
		saveImage(img,color_path,filename)
	else:
		saveImage(img,gray_path,filename)

def sort(img,filename):
	make_path1 = args.output_folder + "yes/"
	make_path2 = args.output_folder + "no/"
	if not os.path.exists(make_path1):
		os.makedirs(make_path1)
	if not os.path.exists(make_path2):
		os.makedirs(make_path2)

	(h, w) = img.shape[:2]
	ratio = h/w

	if(args.exact == True):
		if((ratio >= 1.0) and (h == args.max_size) and (w == args.min_size)):
			path = make_path1
		elif((ratio < 1.0) and (w == args.max_size) and (h == args.min_size)):
			path = make_path1
		else:
			path = make_path2
	else:
		#only works with ratio right now
		if(ratio>=args.min_ratio):
			path = make_path1
		else:
			path = make_path2

	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	else:
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

# def lpipssort(img,filename):

def processImage(img,filename,tag=None):
	if args.process_type == "exclude":	
		exclude(img,filename)
	if args.process_type == "gray_color":	
		gray_color(img,filename)
	if args.process_type == "sort":	
		sort(img,filename)
	if args.process_type == "tagsort":	
		tagsort(img,filename,tag)

def main():
	global args
	global count
	global inter
	args = parse_args()
	count = int(0)
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	if os.path.isdir(args.input_folder):
		print("Processing folder: " + args.input_folder)
	elif os.path.isfile(args.input_folder):
		img = cv2.imread(args.input_folder)
		filename = args.input_folder.split('/')[-1]

		if hasattr(img, 'copy'):
			if(args.verbose): print('processing image: ' + filename)  
			processImage(img,os.path.splitext(filename)[0])
	else:
		print("Not a working input_folder path: " + args.input_folder)
		return;

	for root, subdirs, files in os.walk(args.input_folder):
		if(args.verbose): print('--\nroot = ' + root)

		for subdir in subdirs:
			if(args.verbose): print('\t- subdirectory ' + subdir)

		# sort using LPIPS
		if(args.process_type == "lpips"):
			import lpips

			loss_fn = lpips.LPIPS(net=args.network,version='0.1')			

			img0 = lpips.im2tensor(lpips.load_image(args.start_img))
			
			if not os.path.exists(args.output_folder):
				os.makedirs(args.output_folder)

			if(args.use_gpu):
				loss_fn.cuda()
				img0 = img0.cuda()

			for filename in files:
				file_path = os.path.join(root, filename)
				img1 = lpips.im2tensor(lpips.load_image(file_path))
				
				if(args.use_gpu):
					img1 = img1.cuda()

				dist01 = loss_fn.forward(img0,img1)
				if(args.verbose): print('%s Distance: %.3f'%(filename,dist01))

				if(dist01 <= args.max_dist):
					new_path = os.path.join(args.output_folder, filename)
					shutil.copy2(file_path,new_path)

			continue

		# sort by channel count
		elif(args.process_type=='channels'):
			if not os.path.exists(args.output_folder):
				os.makedirs(args.output_folder)

			gray_path = os.path.join(args.output_folder,'gray')
			if not os.path.exists(gray_path):
				os.makedirs(gray_path)

			rgb_path = os.path.join(args.output_folder,'rgb')
			if not os.path.exists(rgb_path):
				os.makedirs(rgb_path)

			rgba_path = os.path.join(args.output_folder,'rgba')
			if not os.path.exists(rgba_path):
				os.makedirs(rgba_path)

			for filename in files:
				file_path = os.path.join(root, filename)
				img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

				if hasattr(img, 'copy'):
					print(img.shape[-1])
					
					if(img.shape[-1] <= 3): 
						new_path = os.path.join(rgb_path, filename)
						shutil.copy2(file_path,new_path)
					elif(img.shape[-1] == 4):
						new_path = os.path.join(rgba_path, filename)
						shutil.copy2(file_path,new_path)
					else:
						new_path = os.path.join(gray_path, filename)
						shutil.copy2(file_path,new_path)

			continue

        # all other tools
		else:
			for filename in files:
				skipped = False

				file_path = os.path.join(root, filename)
				if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))

				if(args.process_type == "tagsort"):
					import mac_tag

					tags = mac_tag.get(file_path)
					if(len(tags[file_path])>0):
						ts = tags[file_path]
						for t in ts:
							tagpath = os.path.join(args.output_folder, t)
							
							if not os.path.exists(tagpath):
								os.makedirs(tagpath)
							
							new_path = os.path.join(tagpath, filename)
							shutil.copy2(file_path,new_path)

					continue

				if(args.skip_tags != None):
					import mac_tag

					tags = [str(item) for item in args.skip_tags.split(',')]
					# tags = mac_tag.get(file_path)
					# print(tags)
					for tag in tags:
						matches = mac_tag.match(tag,file_path)
						if(file_path in matches):
							print('skipping file: ' + filename)
							new_path = os.path.join(args.output_folder, filename)
							shutil.copy2(file_path,new_path)
							mac_tag.add([tag],[new_path])
							skipped = True
							continue

				if not skipped:
					img = cv2.imread(file_path)

					if hasattr(img, 'copy'):
						processImage(img,filename)
						count = count + int(2)


if __name__ == "__main__":
	main()