import argparse
import os
import shutil

# print(cv2.__version__)

def parse_args():
	desc = "Tools to normalize an image dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-of','--ordered_file', type=str,
		default='rotate',
		help='Process to use. ["rotate","resize","scale"] (default: %(default)s)')

	parser.add_argument('--file_extension', type=str,
		default='png',
		help='file extension ["png","jpg"] (default: %(default)s)')

	args = parser.parse_args()
	return args

def remove(data):
	# print(data)
	filenames = []

	for line in data[1:]:
		cleaned = ' '.join(line.split())
		filenames.append(cleaned.split(' ')[1].split('.')[0])

	print(filenames)

	count = 0
	for filename in filenames:
		file_path = os.path.join(args.input_folder, filename+'.'+args.file_extension)
		os.remove(file_path)

def main():
	global args
	global count
	global inter
	args = parse_args()

	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder)

	# files = [ f for f in os.listdir(args.input_folder) if (not f.startswith('.') and os.path.isfile(os.path.join(args.input_folder, f))) ]

	fileObject = open(args.ordered_file, "r")
	data = fileObject.readlines()

	remove(data)

if __name__ == "__main__":
	main()
