import os
import argparse
import queue
import _thread
import fitz


def parse_args():
    desc = "Script will extract all images found in a pdf or directory of pdfs and save them as pngs."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i', '--input', type=str,
                        default='./input/',
                        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('-o', '--output', type=str,
                        default='./output/',
                        help='Directory path to the outputs folder. (default: %(default)s)')

    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to console.')

    parser.add_argument('-j' '--jobs', type=int,
                        default=1,
                        help='The number of threads to use. (default: %(default)s)')

    return parser.parse_args()


def thread_runner(thread_name, thread_index):
    while not q.empty():
        extract_images(thread_name, q.get())
        if q.empty():
            exit_flags[thread_index] = True


def extract_images(thread_name, pdf_path):
    if args.verbose:
        print('\t%s is extracting images from %s' % (thread_name, pdf_path))

    doc = fitz.open(pdf_path)
    pdf_name = pdf_path.split('/')[-1].split('.')[0].replace(' ', '_')
    for i in range(len(doc)):
        im_list = doc.get_page_images(i)
        for img in im_list:
            xref = img[0]
            pix = fitz.Pixmap(doc, img[0])
            if not pix.n - pix.alpha < 4:  # aka is cmyk
                pix = fitz.Pixmap(fitz.csRGB, pix)
            save_path = os.path.join(args.output, '%s-extracted-%s-%s.png' % (pdf_name, i, xref))
            pix.save(save_path)
            if args.verbose:
                print('\t\t(%s) %s was extracted' % (thread_name, save_path))
            pix = None  # free


def check_ext(path, ext):
    return ext == path.split('.')[-1]


def populate_queue():
    if os.path.isdir(args.input):
        for root, sub_dirs, files in os.walk(args.input):
            if args.verbose:
                print('--\nroot = ' + root)
            for file in files:
                if check_ext(file, 'pdf'):
                    q.put(os.path.join(root, file))
    elif os.path.isfile(args.input) and check_ext(args.input, 'pdf'):
        q.put(args.input)
    else:
        print('Error! No pdf file or directory found at ' + args.input + ' please try again.')
        exit(1)

    if q.empty():
        print('Error! No pdf file or directory found at ' + args.input + ' please try again.')
        exit(1)


def all_exit_flags():
    for ef in exit_flags:
        if not ef:
            return False
    return True

def main():
    global args
    global q
    global exit_flags
    args = parse_args()
    q = queue.Queue()
    exit_flags = []

    populate_queue()

    os.makedirs(args.output, exist_ok=True)

    thread_num = args.j__jobs
    if thread_num > q.qsize():
        print('Warn thread count unnecessary large. Decreasing to %d' % (q.qsize()))
        thread_num = q.qsize()

    for i in range(thread_num):
        try:
            thread_name = 'thread-' + str(i)
            if args.verbose:
                print('starting thread %s' % thread_name)
            exit_flags.append(False)
            _thread.start_new_thread(thread_runner, (thread_name, i))
        except:
            print("Error! unable to start thread.")
            exit(2)

    while not all_exit_flags():
        pass


if __name__ == "__main__":
    main()
