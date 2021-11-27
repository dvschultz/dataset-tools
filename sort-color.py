import argparse
import numpy as np
import os
import cv2
import shutil
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_cie76
from collections import Counter

def parse_args():
	desc = "Tools to normalize an image dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('-v','--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-t','--threshold', type=int,
		default=40,
		help='Color match threshold (default: %(default)s)')
	
	parser.add_argument('--rgb', type=str,
		default=None,
		help='Comma separated RGB value to match against (default: %(default)s')

	parser.add_argument('-c', '--colors', type=str,
		default='red,orange,yellow,green,blue,purple,black,white',
		help='Comma separated list of W3C color keywords to sort by (default: %(default)s')

	args = parser.parse_args()
	return args

# Load the image and convert the colorspace
def get_image(image_path):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image

# Get the most dominant color from the image
# Largely based on the work of Karan Bhanot: 
# https://towardsdatascience.com/color-identification-in-images-machine-learning-application-b26e770c4c71
def get_dominant_color(image):
	# Resize image to help speed up processing
	modified_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
	
	clf = KMeans(n_clusters = 8)
	labels = clf.fit_predict(modified_image)
	
	counts = Counter(labels)
	# sort to ensure correct color percentage
	counts = dict(sorted(counts.items()))
	
	center_colors = clf.cluster_centers_
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]

	dominant_index = max(counts, key=counts.get)

	if(args.verbose): print('Dominant color: ' + str(rgb_colors[dominant_index]))  

	return rgb_colors[dominant_index]

# Pass an image path and if it matches any of the colors
# It will be sorted into that directory
def sort_image_by_color(image): 
	dominant_color = get_dominant_color(get_image(image))
	dominant_lab = rgb2lab(np.uint8(np.asarray([[dominant_color]])))

	for i in range(len(color_list)):
		color_lab = rgb2lab(np.uint8(np.asarray([[color_list[i]]])))
		diff = deltaE_cie76(dominant_lab, color_lab)
		if (diff < args.threshold):
			# If RGB was used, it goes directly in the output directory
			if (args.rgb != None):
				# Copy it there
				if(args.verbose): print('Copying to : ' + args.output_folder)
				shutil.copy2(image, args.output_folder)
			else:
				# Look up the color name
				colors = args.colors.split(',')
				color_path = os.path.join(args.output_folder, colors[i].lower())
				if not os.path.exists(color_path):
					os.makedirs(color_path)
				# Copy it there
				if(args.verbose): print('Copying to : ' + color_path)  
				shutil.copy2(image, color_path)

def main():
	global args
	global color_list
	color_list = []
	args = parse_args()
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	# Color dictionary
	# See: https://en.wikipedia.org/wiki/Web_colors#X11_color_names
	COLORS = {
		'aliceblue': [240,248,255],
		'antiquewhite': [250,235,215],
		'aqua': [0,255,255],
		'aquamarine': [127,255,212],
		'azure': [240,255,255],
		'beige': [245,245,220],
		'bisque': [255,228,196],
		'black': [0,0,0],
		'blanchedalmond': [255,235,205],
		'blue': [0,0,255],
		'blueviolet': [138,43,226],
		'brown': [165,42,42],
		'burlywood': [222,184,135],
		'cadetblue': [95,158,160],
		'chartreuse': [127,255,0],
		'chocolate': [210,105,30],
		'coral': [255,127,80],
		'cornflowerblue': [100,149,237],
		'cornsilk': [255,248,220],
		'crimson': [220,20,60],
		'cyan': [0,255,255],
		'darkblue': [0,0,139],
		'darkcyan': [139,139],
		'darkgoldenrod': [184,134,11],
		'darkgray': [169,169,169],
		'darkgreen': [0,100,0],
		'darkgrey': [169,169,169],
		'darkkhaki': [189,183,107],
		'darkmagenta': [139,0,139],
		'darkolivegreen': [85,107,47],
		'darkorange': [255,140,0],
		'darkorchid': [153,50,204],
		'darkred': [139,0,0],
		'darksalmon': [233,150,122],
		'darkseagreen': [143,188,143],
		'darkslateblue': [72,61,139],
		'darkslategray': [47,79,79],
		'darkslategrey': [47,79,79],
		'darkturquoise': [206,209],
		'darkviolet': [148,0,211],
		'deeppink': [255,20,147],
		'deepskyblue': [0,191,255],
		'dimgray': [105,105,105],
		'dimgrey': [105,105,105],
		'dodgerblue': [30,144,255],
		'firebrick': [178,34,34],
		'floralwhite': [255,250,240],
		'forestgreen': [34,139,34],
		'fuchsia': [255,0,255],
		'gainsboro': [220,220,220],
		'ghostwhite': [248,248,255],
		'gold': [255,215,0],
		'goldenrod': [218,165,32],
		'gray': [128,128,128],
		'green': [0,128,0],
		'greenyellow': [173,255,47],
		'grey': [128,128,128],
		'honeydew': [240,255,240],
		'hotpink': [255,105,180],
		'indianred': [205,92,92],
		'indigo': [75,0,130],
		'ivory': [255,255,240],
		'khaki': [240,230,140],
		'lavender': [230,230,250],
		'lavenderblush': [255,240,245],
		'lawngreen': [124,252,0],
		'lemonchiffon': [255,250,205],
		'lightblue': [173,216,230],
		'lightcoral': [240,128,128],
		'lightcyan': [224,255,255],
		'lightgoldenrodyellow': [250,250,210],
		'lightgray': [211,211,211],
		'lightgreen': [144,238,144],
		'lightgrey': [211,211,211],
		'lightpink': [255,182,193],
		'lightsalmon': [255,160,122],
		'lightseagreen': [32,178,170],
		'lightskyblue': [135,206,250],
		'lightslategray': [119,136,153],
		'lightslategrey': [119,136,153],
		'lightsteelblue': [176,196,222],
		'lightyellow': [255,255,224],
		'lime': [0,255,0],
		'limegreen': [50,205,50],
		'linen': [250,240,230],
		'magenta': [255,0,255],
		'maroon': [128,0,0],
		'mediumaquamarine': [102,205,170],
		'mediumblue': [0,0,205],
		'mediumorchid': [186,85,211],
		'mediumpurple': [147,112,219],
		'mediumseagreen': [60,179,113],
		'mediumslateblue': [123,104,238],
		'mediumspringgreen': [250,154],
		'mediumturquoise': [72,209,204],
		'mediumvioletred': [199,21,133],
		'midnightblue': [25,25,112],
		'mintcream': [245,255,250],
		'mistyrose': [255,228,225],
		'moccasin': [255,228,181],
		'navajowhite': [255,222,173],
		'navy': [0,0,128],
		'oldlace': [253,245,230],
		'olive': [128,128,0],
		'olivedrab': [107,142,35],
		'orange': [255,165,0],
		'orangered': [255,69,0],
		'orchid': [218,112,214],
		'palegoldenrod': [238,232,170],
		'palegreen': [152,251,152],
		'paleturquoise': [175,238,238],
		'palevioletred': [219,112,147],
		'papayawhip': [255,239,213],
		'peachpuff': [255,218,185],
		'peru': [205,133,63],
		'pink': [255,192,203],
		'plum': [221,160,221],
		'powderblue': [176,224,230],
		'purple': [128,0,128],
		'red': [255,0,0],
		'rosybrown': [188,143,143],
		'royalblue': [65,105,225],
		'saddlebrown': [139,69,19],
		'salmon': [250,128,114],
		'sandybrown': [244,164,96],
		'seagreen': [46,139,87],
		'seashell': [255,245,238],
		'sienna': [160,82,45],
		'silver': [192,192,192],
		'skyblue': [135,206,235],
		'slateblue': [106,90,205],
		'slategray': [112,128,144],
		'slategrey': [112,128,144],
		'snow': [255,250,250],
		'springgreen': [255,127],
		'steelblue': [70,130,180],
		'tan': [210,180,140],
		'teal': [0,128,128],
		'thistle': [216,191,216],
		'tomato': [255,99,71],
		'turquoise': [64,224,208],
		'violet': [238,130,238],
		'wheat': [245,222,179],
		'white': [255,255,255],
		'whitesmoke': [245,245,245],
		'yellow': [255,255,0],
		'yellowgreen': [154,205,50]
	}

	# Make sure the output folder has a trailing slash, create it if necessary
	args.output_folder = os.path.join(args.output_folder, '')
	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder)

	# RGB validation
	if (args.rgb != None):
		values = args.rgb.split(',')

		if len(values) != 3:
			print("Invalid number of RGB values: " + str(len(values)))
			return

		for value in values:
			if int(value) not in range(0, 256):
				print("Invalid RGB value: " + value)
				return
		
		color_list = [[int(value) for value in values]]
	
	# Color name validation
	if (args.rgb == None):
		colors = args.colors.split(',')
		for color in colors:
			if color.lower() in COLORS:
				color_list.append(COLORS[color.lower()])
			else:
				print("Invalid color name: " + color)
				return

	# If the input folder is a directory
	if os.path.isdir(args.input_folder):
		print("Processing folder: " + args.input_folder)
	# If the input folder is a file itself
	elif os.path.isfile(args.input_folder):
		if (args.verbose): print('Processing image: ' + args.input_folder)  
		sort_image_by_color(args.input_folder)
		return
	# If it's neither a directory or file, bail
	else:
		print("Not a working input_folder path: " + args.input_folder)
		return

	for root, subdirs, files in os.walk(args.input_folder):
		for filename in files:
			file_path = os.path.join(root, filename)
			# Check to see if it's an image
			img = cv2.imread(file_path)
			# If it is, sort it
			if hasattr(img, 'copy'):
				if (args.verbose): print('Processing image: ' + file_path)  
				sort_image_by_color(file_path)


if __name__ == "__main__":
	main()