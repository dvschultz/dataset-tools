import argparse
import os

import cv2
import numpy as np
import copy

start = False

def parse_args():
    desc = "Interactive tool to generate square crops" 
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-f','--file_extension', type=str,
        default='png',
        help='Border style to use when using the square process type ["png","jpg"] (default: %(default)s)')
    parser.add_argument('-i','--input_folder', type=str,
        default='./input/',
        help='Directory path to the inputs folder. (default: %(default)s)')
    parser.add_argument('--min_size', type=int, 
        default=1024,
        help='Minimum width or height of the cropped images. (default: %(default)s)')
    parser.add_argument('-o','--output_folder', type=str,
        default='./output/',
        help='Directory path to the outputs folder. (default: %(default)s)')
    parser.add_argument('--guides', action='store_true',
        help='Include edge guides')
    parser.add_argument('--padding', type=int, 
        default=0,
        help='Add green borders to image. (default: %(default)s)')
    parser.add_argument('--post', type=str, 
        default=None,
        help='post processing: None, resize (default: %(default)s)')

    args = parser.parse_args()
    return args

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
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)

    # return the resized image
    return resized

def saveImage(img,path,filename):
    print('got: ', filename)
    if(args.file_extension == "png"):
        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif(args.file_extension == "jpg"):
        new_file = os.path.splitext(filename)[0] + ".jpg"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

class Context:
    def __init__(self,imgs,fs):
        self.start = False
        self.counter = 0
        self.i = 0
        self.clean_imgs = copy.deepcopy(imgs)
        self.drawn_imgs = imgs
        self.fs = fs.copy()
        self.xy = (-1,-1)
        self.timer = 0
        self.temp_img = self.drawn_imgs[0]

    def reset(self):
        self.counter = self.counter + 1
        self.start = False
        self.xy = (-1,-1)
        self.timer = 0

    def reset_xy(self):
        print('reset xy')
        self.xy = (-1,-1)
        self.timer = 0

    def check_box(self,d,c0,c1):
        if(d < int(args.min_size/2)):
            return (0,0,255)
        if(min(min(c0,c1)) < 0 ):
            return (0,0,255)
        else:
            return (255,0,0)

    def pad_images(self, pad):
        print('here, pad images: ', pad)
        green = (0,255,0)
        for i, drawn_img in enumerate(self.drawn_imgs):
            self.drawn_imgs[i] = cv2.copyMakeBorder(drawn_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=green)
            self.clean_imgs[i] = cv2.copyMakeBorder(drawn_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=green)
            
    def generate_guides(self,pt):
        red = (0,0,255)
        for drawn_img in self.drawn_imgs:
            (h, w) = drawn_img.shape[:2]
            cv2.line(drawn_img,(pt,0),(pt,h),red,4) #left
            cv2.line(drawn_img,(w-pt,0),(w-pt,h),red,4) #right
            cv2.line(drawn_img,(0,pt),(w,pt),red,4) #top
            cv2.line(drawn_img,(0,h-pt),(w,h-pt),red,4) #bottom

    def mouse(self,event,x,y,flags,param):
        #print(event)
        if(self.start):
            self.temp_img = self.drawn_imgs[self.i].copy()
            d = int(np.abs(np.hypot(x - self.xy[0], y - self.xy[1])))
            c0 = (self.xy[0]-d,self.xy[1]-d)
            c1 = (self.xy[0]+d,self.xy[1]+d)
            print(min(min(c0,c1)))

            color = self.check_box(d,c0,c1)
            cv2.rectangle(self.temp_img,c0,c1,color,6)
            # cv2.imshow('image',tmp)

        if event==4: #CLICK UP
            if self.start == False:
                self.start = not self.start
                self.xy = (x,y)
                print('set x,y: ', self.xy)
            else:
                print(self.xy)
                print(x,y)

                d = int(np.abs(np.hypot(x - self.xy[0], y - self.xy[1])))
                if(d < int(args.min_size/2)): d = int(args.min_size/2)
                y0 = self.xy[1]
                x0 = self.xy[0]
                img = self.clean_imgs[self.i]
                c0 = (self.xy[0]-d,self.xy[1]-d)
                c1 = (self.xy[0]+d,self.xy[1]+d)
                crop = img[y0-d:y0+d,x0-d:x0+d]
                print(min(crop.shape[:2]))

                if(min(crop.shape[:2]) >= args.min_size):
                    fname = self.fs[self.i].split('.')[0] + '_' + str(self.counter)

                    #post processing happens here
                    if(args.post=='resize'):
                        crop = image_resize(crop, max=args.min_size)

                    saveImage(crop,args.output_folder,fname)

                    # cv2.circle(self.drawn_imgs[self.i],self.xy,d,(255,0,0),10)
                    cv2.rectangle(self.drawn_imgs[self.i],c0,c1,(0,255,0),10)
                else:
                    cv2.rectangle(self.drawn_imgs[self.i],c0,c1,(0,0,255),10)


                self.reset()


def interactive(imgs,fs):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)

    c = Context(imgs,fs)

    cv2.imshow('image',c.drawn_imgs[c.i])
    cv2.resizeWindow('image', 1200,800)
    cv2.moveWindow('image', 0,0)
    cv2.setMouseCallback('image', c.mouse)

    print(args.padding)
    if (args.padding > 0):
        c.pad_images(args.padding)

    if(args.guides):
        c.generate_guides(int(args.min_size/2))

    while(1):
        if(c.start):
            cv2.imshow('image',c.temp_img)
        else:
            cv2.imshow('image',c.drawn_imgs[c.i])
        c.timer+=1

        k = cv2.waitKey(33)
        if(k==-1):
            continue
        elif(k==27):
            cv2.destroyAllWindows()
        elif(k==124 or k==32): #space bar
            c.i+=1
            print(c.i)
            print('next image: ', c.fs[c.i])
            c.reset_xy()
            c.temp_img = c.drawn_imgs[c.i]
            
            if(c.i >= len(imgs)):
                cv2.destroyWindow('image')
                
        elif(k==123 or k==108): # l key
            c.i-=1
            if(c.i < 0): c.i = 0
            c.reset_xy()
            c.temp_img = c.drawn_imgs[c.i]
            print('prev image: ', fs[c.i])
            
        else:
            print('pressed: ', k)

def main():
    global args
    args = parse_args()

    os.environ['OPENCV_IO_ENABLE_JASPER']= "true"
    inter = cv2.INTER_CUBIC

    global imgs, fs
    imgs = []
    fs = []

    if os.path.isdir(args.input_folder):
        print("Processing folder: " + args.input_folder)
    elif os.path.isfile(args.input_folder):
        img = cv2.imread(args.input_folder)
        filename = args.input_folder.split('/')[-1]

        # if hasattr(img, 'copy'):
        #     if(args.verbose): print('processing image: ' + filename)  
        #     processImage(img,os.path.splitext(filename)[0])
    else:
        print("Not a working input_folder path: " + args.input_folder)
        return;

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for root, subdirs, files in os.walk(args.input_folder):
        print('--\nroot = ' + root)

        for subdir in subdirs:
            print('\t- subdirectory ' + subdir)

        for filename in files:
            file_path = os.path.join(root, filename)
            img = cv2.imread(file_path)

            if hasattr(img, 'copy'):
                fs.append(filename)
                imgs.append(img)  

    interactive(imgs,fs)
	

if __name__ == "__main__":
    main()