import argparse
import numpy as np
import scipy.ndimage as pyimg
import os
import cv2
import random
from concurrent.futures import ThreadPoolExecutor

def parse_args():
	desc = "Tools to normalize an image dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('--workers', type=int,
		default=0,
		help='Parallel worker threads for processing files. 0 = auto (all cores). 1 = serial. (default: %(default)s)')

	parser.add_argument('--png_compression', type=int,
		default=0,
		help='PNG compression level 0-9. 0 = none/fastest CPU but largest files; higher = smaller files, less disk I/O. (default: %(default)s)')

	parser.add_argument('--jpg_quality', type=int,
		default=90,
		help='JPEG quality 0-100 for jpg output. (default: %(default)s)')

	parser.add_argument('--force_max', action='store_true',
		help='Force max size')

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-p','--process_type', type=str,
		default='resize',
		help='Process to use. ["resize","square","crop_to_square","canny","canny-pix2pix","crop_square_patch","scale","many_squares","crop","distance"] (default: %(default)s)')

	parser.add_argument('--blur_type', type=str,
		default='gaussian',
		help='Blur process to use. Use with --process_type canny. ["none","gaussian","median"] (default: %(default)s)')

	parser.add_argument('--blur_amount', type=int, 
		default=1,
		help='Amount of blur to apply (use odd numbers). Use with --blur_type.  (default: %(default)s)')


	parser.add_argument('-cmin','--canny_min', type=int, 
		default=100,
		help='Minimum threshold for Canny use.  (default: %(default)s)')


	parser.add_argument('-cmax','--canny_max', type=int, 
		default=300,
		help='Maximum threshold for Canny use.  (default: %(default)s)')

	parser.add_argument('--max_size', type=int, 
		default=1024,
		help='Maximum width or height of the output images. (default: %(default)s)')

	parser.add_argument('--height', type=int, 
		default=None,
		help='Maximum height of the output image (for use with --process_type crop). (default: %(default)s)')

	parser.add_argument('--width', type=int, 
		default=None,
		help='Maximum width of output image (for use with --process_type crop). (default: %(default)s)')

	parser.add_argument('--shift_y', type=int, 
		default=0,
		help='y coordinate shift (for use with --process_type crop). (default: %(default)s)')

	parser.add_argument('--v_align', type=str, 
		default='center',
		help='-vertical alignment options: top, bottom, center (for use with --process_type crop_to_square). (default: %(default)s)')

	parser.add_argument('--h_align', type=str, 
		default='center',
		help='-vertical alignment options: left, right, center (for use with --process_type crop_to_square). (default: %(default)s)')

	parser.add_argument('--shift_x', type=int, 
		default=0,
		help='x coordinate shift (for use with --process_type crop). (default: %(default)s)')

	parser.add_argument('--scale', type=float, 
		default=2.0,
		help='Scalar value. For use with scale process type (default: %(default)s)')

	parser.add_argument('--skip_tags', type=str, 
        default=None,
        help='comma separated color tags (for Mac only) (default: %(default)s)')

	parser.add_argument('--direction', type=str,
		default='AtoB',
		help='Paired Direction. For use with pix2pix process. ["AtoB","BtoA"] (default: %(default)s)')

	parser.add_argument('--border_type', type=str,
		default='stretch',
		help='Border style to use when using the square process type ["stretch","reflect","solid","inpaint"] (default: %(default)s)')

	parser.add_argument('--border_color', type=str,
		default='255,255,255',
		help='border color to use with the `solid` border type; use bgr values (default: %(default)s)')

	parser.add_argument('--mirror', action='store_true',
		help='Adds mirror augmentation.')

	parser.add_argument('--rotate', action='store_true',
		help='Adds 90 degree rotation augmentation.')

	parser.add_argument('-f','--file_extension', type=str,
		default='png',
		help='Border style to use when using the square process type ["png","jpg"] (default: %(default)s)')

	feature_parser = parser.add_mutually_exclusive_group(required=False)
	feature_parser.add_argument('--keep_name', dest='name', action='store_true')
	feature_parser.add_argument('--numbered', dest='name', action='store_false')
	parser.set_defaults(name=True)

	args = parser.parse_args()
	return args

def ensure_output_dir(path):
	"""Create output directory if it doesn't exist (thread-safe)."""
	os.makedirs(path, exist_ok=True)

def pick_interp(src_shape, dim):
	"""Choose the fastest/highest-quality interpolation for a resize.

	INTER_AREA is faster and sharper when shrinking; fall back to the
	global `inter` (INTER_CUBIC) when upscaling.
	"""
	(h, w) = src_shape[:2]
	if dim[0] * dim[1] < w * h:
		return cv2.INTER_AREA
	return inter

def get_border_type():
	"""Get OpenCV border type from args.border_type."""
	if args.border_type == 'solid':
		return cv2.BORDER_CONSTANT
	elif args.border_type == 'reflect':
		return cv2.BORDER_REFLECT
	else:
		return cv2.BORDER_REPLICATE

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
    resized = cv2.resize(image, dim, interpolation = pick_interp(image.shape, dim))

    # return the resized image
    return resized

def saveImage(img, path, filename):
	"""Save an image to disk with the specified file extension.

	Args:
		img: OpenCV image array to save
		path: Directory path where the image will be saved
		filename: Base filename (extension will be replaced based on args.file_extension)
	"""
	if(args.file_extension == "png"):
		new_file = os.path.splitext(filename)[0] + ".png"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])
	elif(args.file_extension == "jpg"):
		new_file = os.path.splitext(filename)[0] + ".jpg"
		cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])

def image_scale(image, scalar = 1.0):
	"""Scale an image by a scalar factor.

	Args:
		image: OpenCV image array
		scalar: Scaling factor (default: 1.0)

	Returns:
		Scaled OpenCV image array
	"""
	(h, w) = image.shape[:2]
	dim = (int(w*scalar),int(h*scalar))
	resized = cv2.resize(image, dim, interpolation = pick_interp(image.shape, dim))
	return resized

def arbitrary_crop(img, h_crop,w_crop):
	error = False
	(h, w) = img.shape[:2]
	if(h>h_crop):
		hdiff = int((h-h_crop)/2) + args.shift_y

		if( ((hdiff+h_crop) > h) or (hdiff < 0)):
			print("error! crop settings are too much for this image")
			error = True
		else:
			img = img[hdiff:hdiff+h_crop,0:w]
	if(w>w_crop):
		wdiff = int((w-w_crop)/2) + args.shift_x
		
		if( ((wdiff+w_crop) > w) or (wdiff < 0) ):
			print("error! crop settings are too much for this image")
			error = True
		else:
			img = img[0:h_crop,wdiff:wdiff+w_crop]
	return img, error

def crop_to_square(img):
	"""Crop an image to a square by removing edges based on alignment settings.

	Args:
		img: OpenCV image array

	Returns:
		Square-cropped OpenCV image array
	"""
	(h, w) = img.shape[:2]

	cropped = img.copy()
	if w > h:
		if (args.h_align=='left'):
			cropped = img[:h,:h]
		elif (args.h_align=='right'):
			cropped = img[0:h, w-h:w]
		else:
			diff = int((w-h)/2)
			cropped = img[0:h, diff:diff+h]
	elif h > w:
		if (args.v_align=='top'):
			cropped = img[:w, :w]
		elif (args.v_align=='bottom'):
			cropped = img[h-w:h, 0:w]
		else:
			diff = int((h-w)/2)
			cropped = img[diff:diff+w, 0:w]

	return cropped

def crop_square_patch(img, imgSize):
	"""Extract a random square patch from an image.

	Args:
		img: OpenCV image array
		imgSize: Size of the square patch to extract

	Returns:
		Square patch of the specified size
	"""
	(h, w) = img.shape[:2]

	rH = random.randint(0,h-imgSize)
	rW = random.randint(0,w-imgSize)
	cropped = img[rH:rH+imgSize,rW:rW+imgSize]

	return cropped

def processCanny(img):
	"""Apply Canny edge detection with optional blur preprocessing.

	Args:
		img: OpenCV image array (BGR color)

	Returns:
		Grayscale edge-detected image
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	if(args.blur_type=='gaussian'):
		gray = cv2.GaussianBlur(gray, (args.blur_amount, args.blur_amount), 0)
	elif(args.blur_type=='median'):
		gray = cv2.medianBlur(gray,args.blur_amount)
	gray = cv2.Canny(gray,args.canny_min,args.canny_max)

	return gray

def makeResize(img,filename,scale):
	remakePath = args.output_folder + str(scale)+"/"
	ensure_output_dir(remakePath)

	# cv2.resize / image_resize return a fresh array, so no defensive copy is
	# needed here (saves a full-image memcpy per file).
	if(args.height!=None and args.width!=None):
		dim = (args.width, args.height)
		img_copy = cv2.resize(img, dim, interpolation = pick_interp(img.shape, dim))
	else:
		img_copy = image_resize(img, max = scale)

	saveImage(img_copy,remakePath,filename)
	if (args.mirror): flipImage(img_copy,filename,remakePath)
	if (args.rotate): rotateImage(img_copy,filename,remakePath)

def makeDistance(img,filename,scale):
	makePath = args.output_folder + "distance-"+ str(args.max_size)+"/"
	ensure_output_dir(makePath)

	# image_resize returns a fresh array; the later in-place channel writes
	# operate on that, so the source img is never mutated. No copy needed.
	img_copy = image_resize(img, max = scale)

	BW = img_copy[:,:,0] > 127
	G_channel = pyimg.distance_transform_edt(BW)
	G_channel[G_channel>32]=32
	B_channel = pyimg.distance_transform_edt(1-BW)
	B_channel[B_channel>200]=200
	img_copy[:,:,1] = G_channel.astype('uint8')
	img_copy[:,:,0] = B_channel.astype('uint8')

	saveImage(img_copy,makePath,filename)
	if (args.mirror): flipImage(img_copy,filename,makePath)
	if (args.rotate): rotateImage(img_copy,filename,makePath)

def makeScale(img,filename,scale):
	remakePath = args.output_folder + "scale_"+str(scale)+"/"
	ensure_output_dir(remakePath)

	img_copy = image_scale(img, scale)

	saveImage(img_copy,remakePath,filename)
	if (args.mirror): flipImage(img_copy,filename,remakePath)
	if (args.rotate): rotateImage(img_copy,filename,remakePath)

# https://docs.opencv.org/4.5.2/df/d3d/tutorial_py_inpainting.html
def inpaintSquare(img,scale):
	(h, w) = img.shape[:2]
	mask = np.zeros((h,w,1), np.uint8)
	diff_x = scale - w
	diff_y = scale - h

	if(diff_x%2 == 0 and diff_y%2 == 0 ):
		img = cv2.copyMakeBorder(img, int(diff_y/2), int(diff_y/2), int(diff_x/2), int(diff_x/2), cv2.BORDER_CONSTANT,value=[255,255,255])
		mask = cv2.copyMakeBorder(mask, int(diff_y/2), int(diff_y/2), int(diff_x/2), int(diff_x/2), cv2.BORDER_CONSTANT,value=[255,255,255])
	elif(diff_x%2 == 0):
		img = cv2.copyMakeBorder(img, int(diff_y/2)+1, int(diff_y/2), int(diff_x/2), int(diff_x/2), cv2.BORDER_CONSTANT,value=[255,255,255])
		mask = cv2.copyMakeBorder(mask, int(diff_y/2)+1, int(diff_y/2), int(diff_x/2), int(diff_x/2), cv2.BORDER_CONSTANT,value=[255,255,255])
	else:
		img = cv2.copyMakeBorder(img, int(diff_y/2), int(diff_y/2), int(diff_x/2)+1, int(diff_x/2), cv2.BORDER_CONSTANT,value=[255,255,255])
		mask = cv2.copyMakeBorder(mask, int(diff_y/2), int(diff_y/2), int(diff_x/2)+1, int(diff_x/2), cv2.BORDER_CONSTANT,value=[255,255,255])

	return cv2.inpaint(img,mask,3,cv2.INPAINT_NS)


def makeSquare(img,filename,scale):
	sqPath = args.output_folder + "sq-"+str(scale)+"/"
	ensure_output_dir(sqPath)

	# No copy: every branch below (image_resize / copyMakeBorder / inpaintSquare)
	# returns a new array rather than mutating the source in place.
	img_sq = img
	(h, w) = img_sq.shape[:2]
	if((h < scale) and (w < scale)):
		if(args.verbose): print('skip resize')
	else:
		img_sq = image_resize(img_sq, max = scale)

	if(args.border_type=="inpaint"):
		img_sq = inpaintSquare(img_sq, scale)
	else:
		bType = get_border_type()
		bColor = [int(item) for item in args.border_color.split(',')]

		(h, w) = img_sq.shape[:2]
		if(args.force_max):
			diff_x = scale - w
			diff_y = scale - h

			if(diff_x%2 == 0 and diff_y%2 == 0 ):
				img_sq = cv2.copyMakeBorder(img_sq, int(diff_y/2), int(diff_y/2), int(diff_x/2), int(diff_x/2), bType,value=bColor)
			elif(diff_x%2 == 0):
				img_sq = cv2.copyMakeBorder(img_sq, int(diff_y/2)+1, int(diff_y/2), int(diff_x/2), int(diff_x/2), bType,value=bColor)
			elif(diff_y%2 == 0):
				img_sq = cv2.copyMakeBorder(img_sq, int(diff_y/2), int(diff_y/2), int(diff_x/2)+1, int(diff_x/2), bType,value=bColor)
			
			#checks if our division leads to incorrect image sizes
			elif ((h + (int(diff_y/2)*2) != scale) or (w + (int(diff_x/2)*2) != scale)):
				
				if(h + (int(diff_y/2)*2) != scale) and (w + (int(diff_x/2)*2) != scale):
					img_sq = cv2.copyMakeBorder(img_sq, int(diff_y/2)+1, int(diff_y/2), int(diff_x/2)+1, int(diff_x/2), bType,value=bColor)
				elif(h + (int(diff_y/2)*2) != scale):
					img_sq = cv2.copyMakeBorder(img_sq, int(diff_y/2), int(diff_y/2), int(diff_x/2)+1, int(diff_x/2), bType,value=bColor)
				elif(w + (int(diff_x/2)*2) != scale):
					img_sq = cv2.copyMakeBorder(img_sq, int(diff_y/2)+1, int(diff_y/2), int(diff_x/2), int(diff_x/2), bType,value=bColor)	
			else:
				img_sq = cv2.copyMakeBorder(img_sq, int(diff_y/2), int(diff_y/2), int(diff_x/2)+1, int(diff_x/2), bType,value=bColor)
		elif(h > w):
			# pad left/right
			diff = h-w
			if(diff%2 == 0):
				img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff/2), int(diff/2), bType,value=bColor)
			else:
				img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff/2)+1, int(diff/2), bType,value=bColor)
		elif(w > h):
			# pad top/bottom
			diff = w-h

			if(args.v_align == 'bottom'):
				img_sq = cv2.copyMakeBorder(img_sq, diff, 0, 0, 0, bType,value=bColor)
			else:
				if(diff%2 == 0):
					img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2), 0, 0, bType,value=bColor)
				else:
					img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2)+1, 0, 0, bType,value=bColor)
		else:
			diff = scale-h
			if(diff%2 == 0):
				img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2), int(diff/2), int(diff/2), bType,value=bColor)
			else:
				img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2)+1, int(diff/2), int(diff/2)+1, bType,value=bColor)

	saveImage(img_sq,sqPath,filename)
	if (args.mirror): flipImage(img_sq,filename,sqPath)
	if (args.rotate): rotateImage(img_sq,filename,sqPath)

	
	

def makeCanny(img,filename,scale):
	make_path = args.output_folder + "canny-"+str(scale)+"/"
	ensure_output_dir(make_path)

	img_copy = image_resize(img, max = scale)
	gray = processCanny(img_copy)

	saveImage(gray,make_path,filename)
	if (args.mirror): flipImage(gray,filename,make_path)
	if (args.rotate): rotateImage(gray,filename,make_path)

def makeCrop(img,filename):
	make_path = args.output_folder + "crop-"+str(args.height)+"x"+str(args.width)+"/"
	ensure_output_dir(make_path)

	img_copy,error = arbitrary_crop(img,args.height,args.width)

	if((img_copy.shape[0] != args.height) or (img_copy.shape[1] != args.width)):
		# print(img_copy.shape[1], args.height, img_copy.shape[0], args.width)
		print("unable to crop to the size requested")

	if (error==False):
		saveImage(img_copy,make_path,filename)
		if (args.mirror): flipImage(img_copy,filename,make_path)
		if (args.rotate): rotateImage(img_copy,filename,make_path)
	else:
		if(args.verbose): print(filename+" returned an error")

def makeSquareCrop(img,filename,scale):
	make_path = args.output_folder + "sq-"+str(scale)+"/"
	ensure_output_dir(make_path)

	img_copy = crop_to_square(img)
	img_copy = image_resize(img_copy, max = scale)

	saveImage(img_copy,make_path,filename)
	if (args.mirror): flipImage(img_copy,filename,make_path)
	if (args.rotate): rotateImage(img_copy,filename,make_path)

def makeManySquares(img,filename,scale):
	make_path = args.output_folder + "many_squares-"+str(scale)+"/"
	ensure_output_dir(make_path)

	# Crops below are slice views and each is resized into a new array before
	# saving, so the source img is never mutated. No copy needed.
	img_copy = img
	(h, w) = img_copy.shape[:2]
	img_ratio = h/w

	if(img_ratio >= 1.2):

		#crop images from top and bottom
		crop = img_copy[0:w,0:w]
		crop = image_resize(crop, max = scale)
		saveImage(crop,make_path,filename+"-1")
		if (args.mirror): flipImage(crop,filename+"-1",make_path)
		if (args.rotate): rotateImage(crop,filename+"-1",make_path)

		crop = img_copy[h-w:h,0:w]
		crop = image_resize(crop, max = scale)
		saveImage(crop,make_path,filename+"-2")
		if (args.mirror): flipImage(crop,filename+"-2",make_path)
		if (args.rotate): rotateImage(crop,filename+"-2",make_path)

	elif(img_ratio <= .8):
		#crop images from left and right
		if(args.verbose): print(os.path.splitext(filename)[0] + ': wide image')

		crop = img_copy[0:h,0:h]
		crop = image_resize(crop, max = scale)
		saveImage(crop,make_path,filename+"-1")
		if (args.mirror): flipImage(crop,filename+"-1",make_path)
		if (args.rotate): rotateImage(crop,filename+"-1",make_path)

		crop = img_copy[0:h,w-h:w]
		crop = image_resize(crop, max = scale)
		saveImage(crop,make_path,filename+"-2")
		if (args.mirror): flipImage(crop,filename+"-2",make_path)
		if (args.rotate): rotateImage(crop,filename+"-2",make_path)

	else:
		img_copy = crop_to_square(img_copy)
		img_copy = image_resize(img_copy, max = scale)
		saveImage(img_copy,make_path,filename)
		
		if(args.mirror): flipImage(img_copy,filename,make_path)
		if(args.rotate): rotateImage(img_copy,filename,make_path)
		

def makeSquareCropPatch(img,filename,scale):
	make_path = args.output_folder + "sq-"+str(scale)+"/"
	ensure_output_dir(make_path)

	img_copy = crop_square_patch(img,args.max_size)

	saveImage(img_copy,make_path,filename)
	if (args.mirror): flipImage(img_copy,filename,make_path)
	if (args.rotate): rotateImage(img_copy,filename,make_path)

def makePix2Pix(img,filename,scale,direction="BtoA",value=[0,0,0]):
	img_p2p = image_resize(img, max = scale)
	(h, w) = img_p2p.shape[:2]
	bType = cv2.BORDER_CONSTANT

	make_path = args.output_folder + "pix2pix-"+str(h)+"/"
	ensure_output_dir(make_path)

	canny = cv2.cvtColor(processCanny(img_p2p),cv2.COLOR_GRAY2RGB)

	if(direction=="BtoA"):
		img_p2p = cv2.copyMakeBorder(img_p2p, 0, 0, w, 0, bType, None, value)
		img_p2p[0:h,0:w] = canny

	saveImage(img_p2p,make_path,filename)

def flipImage(img, filename, path):
	"""Create and save a horizontally flipped version of an image.

	Args:
		img: OpenCV image array
		filename: Base filename for the output
		path: Directory path where the flipped image will be saved
	"""
	flip_img = cv2.flip(img, 1)
	flip_file = os.path.splitext(filename)[0] + "-flipped"
	saveImage(flip_img,path,flip_file)

def rotateImage(img, filename, path):
	"""Create and save 90, 180, and 270 degree rotated versions of an image.

	Args:
		img: OpenCV image array
		filename: Base filename for the output
		path: Directory path where the rotated images will be saved
	"""
	# cv2.rotate does exact 90-degree pixel permutations: faster and lossless
	# compared to imutils.rotate_bound (which uses interpolated warpAffine).
	saveImage(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),path,filename+"-rot90")
	saveImage(cv2.rotate(img, cv2.ROTATE_180),path,filename+"-rot180")
	saveImage(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),path,filename+"-rot270")
	
def processImage(img, filename):
	"""Process an image based on the selected process type.

	Args:
		img: OpenCV image array
		filename: Base filename for output files
	"""
	if args.process_type == "resize":
		makeResize(img,filename,args.max_size)
	elif args.process_type == "square":
		makeSquare(img,filename,args.max_size)
	elif args.process_type == "crop_to_square":
		makeSquareCrop(img,filename,args.max_size)
	elif args.process_type == "canny":
		makeCanny(img,filename,args.max_size)
	elif args.process_type == "canny-pix2pix":
		makePix2Pix(img,filename,args.max_size)
	elif args.process_type == "crop_square_patch":
		makeSquareCropPatch(img,filename,args.max_size)
	elif args.process_type == "scale":
		makeScale(img,filename,args.scale)
	elif args.process_type == "many_squares":
		makeManySquares(img,filename,args.max_size)
	elif args.process_type == "crop":
		makeCrop(img,filename)
	elif args.process_type == "distance":
		makeDistance(img,filename,args.max_size)

def process_file(file_path, out_name):
	"""Read one image and run the selected process. Safe to call from a thread:
	OpenCV's imread/resize/imwrite release the GIL, so this parallelizes well."""
	img = cv2.imread(file_path)
	if hasattr(img, 'copy'):
		if(args.verbose): print('processing image: ' + os.path.basename(file_path))
		processImage(img, out_name)

def main():
	global args
	global inter
	args = parse_args()
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	if(args.skip_tags != None):
		import mac_tag

	if os.path.isdir(args.input_folder):
		print("Processing folder: " + args.input_folder)
	elif os.path.isfile(args.input_folder):
		filename = os.path.basename(args.input_folder)
		process_file(args.input_folder, os.path.splitext(filename)[0])

	# Enumerate files serially (deterministic order + thread-safe tagging /
	# numbered naming), then process them in parallel.
	tasks = []
	count = 0
	for root, subdirs, files in os.walk(args.input_folder):
		if(args.verbose): print('--\nroot = ' + root)

		for subdir in subdirs:
			if(args.verbose): print('\t- subdirectory ' + subdir)

		for filename in files:
			if filename.startswith('.'):
				continue
			file_path = os.path.join(root, filename)
			if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))

			if(args.skip_tags != None):
				skipped = False
				tags = [str(item) for item in args.skip_tags.split(',')]
				for tag in tags:
					matches = mac_tag.match(tag,file_path)
					if(file_path in matches):
						print('skipping file: ' + filename)
						skipped = True
				if skipped:
					continue

			out_name = os.path.splitext(filename)[0] if args.name else str(count)
			count += 1
			tasks.append((file_path, out_name))

	workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
	if workers == 1 or len(tasks) <= 1:
		for file_path, out_name in tasks:
			process_file(file_path, out_name)
	else:
		with ThreadPoolExecutor(max_workers=workers) as pool:
			# list() forces iteration so exceptions surface instead of being swallowed
			list(pool.map(lambda t: process_file(*t), tasks))


if __name__ == "__main__":
	main()
