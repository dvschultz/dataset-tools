import argparse
import os
from psd_tools import PSDImage

def parse_args():
	desc = "Tools to normalize an image dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-v','--verbose', action='store_true',
		help='Print progress to console.')

	args = parser.parse_args()
	return args

def main():
    global args
    global inter

    args = parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for root, subdirs, files in os.walk(args.input_folder):
        files = [f for f in files if not f[0] == '.']
        if(args.verbose): print('--\nroot = ' + root)

        for subdir in subdirs:
            if(args.verbose): print('\t- subdirectory ' + subdir)

        for filename in files:
            fn = os.path.splitext(filename)[0]
            file_path = os.path.join(root, filename)
            out_path = os.path.join(args.output_folder, (fn+'.png'))
            if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))

            psd = PSDImage.load(file_path)
            psd.print_tree()
            psd.as_PIL().save(out_path)


if __name__ == "__main__":
    main()