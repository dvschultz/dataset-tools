import argparse
import os

import cv2
import numpy as np
import copy

from utils.load_images import load_images, load_images_multi_thread

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
    parser.add_argument('-m','--mode', type=str,
        default='center',
        help='Mode type. Options: center, bilateral. (default: %(default)s)')
    parser.add_argument('-o','--output_folder', type=str,
        default='./output/',
        help='Directory path to the outputs folder. (default: %(default)s)')
    parser.add_argument('--guides', action='store_true',
        help='Include edge guides')
    parser.add_argument('--padding', type=int,
        default=0,
        help='Add green borders to image. (default: %(default)s)')
    parser.add_argument('--outpaint', type=int,
        default=0,
        help='Extend image with data from neighboring pixels. (default: %(default)s)')
    parser.add_argument('--choose', action='store_true',
        help='classify each image as yes (y) or no (n), copying into /yes/ or /no/ accordingly')
    parser.add_argument('--post', type=str,
        default=None,
        help='post processing: None, resize (default: %(default)s)')
    parser.add_argument('-j' '--jobs', type=int,
        default=1,
        help='The number of threads to use. (default: %(default)s)')
    parser.add_argument('--verbose', action='store_true',
        help='Print progress to console.')

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

def outpaint_image(in_img, pad_sz):

    in_img_h, in_img_w, channels = in_img.shape

    out_img_h = in_img_h + (pad_sz * 2)
    out_img_w = in_img_w + (pad_sz * 2)

    mask = np.zeros((out_img_h, out_img_w ,1), np.uint8)

    fill_color = (255,255,255)

    out_img = cv2.copyMakeBorder(in_img, pad_sz, pad_sz, pad_sz, pad_sz, cv2.BORDER_CONSTANT, value=fill_color)

    mask = cv2.rectangle(mask,(0,0),(pad_sz-1, out_img_h-1),(255),-1)
    mask = cv2.rectangle(mask,(0,0),(out_img_w-1, pad_sz-1),(255),-1)
    mask = cv2.rectangle(mask,(0, out_img_h-pad_sz),(out_img_w-1, out_img_h-1),(255),-1)
    mask = cv2.rectangle(mask,(out_img_w-pad_sz,0),(out_img_w-1, out_img_h-1),(255),-1)

    return cv2.inpaint(out_img,mask,3,cv2.INPAINT_TELEA)


def saveImage(img,path,filename):
    #print('got: ', filename)
    if(args.file_extension == "png"):
        new_file = os.path.splitext(filename)[0] + ".png"
        image_write_path = os.path.join(path, new_file)
        cv2.imwrite(image_write_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif(args.file_extension == "jpg"):
        new_file = os.path.splitext(filename)[0] + ".jpg"
        image_write_path = os.path.join(path, new_file)
        cv2.imwrite(image_write_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"saving to {image_write_path}?")
class Context:
    def __init__(self,imgs,fs,mode):
        self.start = False
        self.clicks = 0
        self.mode = mode
        self.counter = 0
        self.i = 0
        self.clean_imgs = copy.deepcopy(imgs)
        self.drawn_imgs = imgs
        self.fs = fs.copy()
        self.xy = (-1,-1)
        self.b_xy = []
        self.timer = 0
        self.temp_img = self.drawn_imgs[0]
        self.a = 0

    def reset(self):
        self.counter = self.counter + 1
        self.start = False
        self.xy = (-1,-1)
        self.timer = 0
        self.b_xy = []
        self.a = 0

    def reset_xy(self):
        print('reset xy')
        self.start = False
        self.xy = (-1,-1)
        self.timer = 0
        self.b_xy = []

    def switch_mode(self):
        if(self.mode == "center"):
            self.mode = "bilateral"
        elif(self.mode == "bilateral"):
            self.mode = "center"
        else:
            print("mode not set")

        self.reset_xy
        print(self.mode)

    def check_box(self,d,c0,c1):
        if(d < int(args.min_size/2)):
            return (0,0,255)
        if(min(min(c0,c1)) < 0 ):
            return (0,0,255)
        else:
            return (255,0,0)

    def draw_rotated_box(self,image,d,color):
        # cv2.line(self.temp_img,(self.b_xy[0]),(self.b_xy[1]),blue,3)
        # cv2.circle(self.temp_img,self.xy,d,color,3)

        pt0 = (int(self.xy[0]-d), int(self.xy[1]-d)) #top left
        pt1 = (int(self.xy[0]+d), int(self.xy[1]-d)) #top right
        pt2 = (int(self.xy[0]+d), int(self.xy[1]+d)) #bottom right
        pt3 = (int(self.xy[0]-d), int(self.xy[1]+d)) #bottom left

        theta = (self.a * np.pi / 180)
        rotated_x = np.cos(theta) * (pt0[0] - self.xy[0]) - np.sin(theta) * (pt0[1] - self.xy[1]) + self.xy[0]
        rotated_y = np.sin(theta) * (pt0[0] - self.xy[0]) + np.cos(theta) * (pt0[1] - self.xy[1]) + self.xy[1]
        point_0 = (int(rotated_x), int(rotated_y))

        # Point 1
        rotated_x = np.cos(theta) * (pt1[0] - self.xy[0]) - np.sin(theta) * (pt1[1] - self.xy[1]) + self.xy[0]
        rotated_y = np.sin(theta) * (pt1[0] - self.xy[0]) + np.cos(theta) * (pt1[1] - self.xy[1]) + self.xy[1]
        point_1 = (int(rotated_x), int(rotated_y))

        # Point 2
        rotated_x = np.cos(theta) * (pt2[0] - self.xy[0]) - np.sin(theta) * (pt2[1] - self.xy[1]) + self.xy[0]
        rotated_y = np.sin(theta) * (pt2[0] - self.xy[0]) + np.cos(theta) * (pt2[1] - self.xy[1]) + self.xy[1]
        point_2 = (int(rotated_x), int(rotated_y))

        # Point 3
        rotated_x = np.cos(theta) * (pt3[0] - self.xy[0]) - np.sin(theta) * (pt3[1] - self.xy[1]) + self.xy[0]
        rotated_y = np.sin(theta) * (pt3[0] - self.xy[0]) + np.cos(theta) * (pt3[1] - self.xy[1]) + self.xy[1]
        point_3 = (int(rotated_x), int(rotated_y))

        pts = np.array([point_0, point_1, point_2, point_3], np.int32)
        cv2.polylines(image, [pts], True, color, 6)

    def pad_images(self, pad):

        green = (0,255,0)
        for i, drawn_img in enumerate(self.drawn_imgs):
            self.drawn_imgs[i] = cv2.copyMakeBorder(drawn_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=green)
            self.clean_imgs[i] = cv2.copyMakeBorder(drawn_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=green)

    def outpaint(self, pad):
        for i, drawn_img in enumerate(self.drawn_imgs):
            self.drawn_imgs[i] = outpaint_image(drawn_img, pad)
            self.clean_imgs[i] = np.copy(self.drawn_imgs[i])


    def generate_guides(self,pt):
        red = (0,0,255)
        for drawn_img in self.drawn_imgs:
            (h, w) = drawn_img.shape[:2]
            cv2.line(drawn_img,(pt,0),(pt,h),red,4) #left
            cv2.line(drawn_img,(w-pt,0),(w-pt,h),red,4) #right
            cv2.line(drawn_img,(0,pt),(w,pt),red,4) #top
            cv2.line(drawn_img,(0,h-pt),(w,h-pt),red,4) #bottom

    def make_crop(self,d):
        img = self.clean_imgs[self.i]

        if(self.a != 0):
            rot_mat = cv2.getRotationMatrix2D(self.xy, self.a, 1.0)
            img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        y0 = self.xy[1]
        x0 = self.xy[0]
        c0 = (self.xy[0]-d,self.xy[1]-d)
        c1 = (self.xy[0]+d,self.xy[1]+d)
        crop = img[y0-d:y0+d,x0-d:x0+d]
        # print(min(crop.shape[:2]))

        if(min(crop.shape[:2]) >= args.min_size):
            fname = self.fs[self.i].split('.')[0] + '_' + str(self.counter)

            #post processing happens here
            if(args.post=='resize'):
                crop = image_resize(crop, max=args.min_size)

            saveImage(crop,args.output_folder,fname)

            # cv2.circle(self.drawn_imgs[self.i],self.xy,d,(255,0,0),10)
            if(self.a == 0):
                cv2.rectangle(self.drawn_imgs[self.i],c0,c1,(0,255,0),10)
            else:
                self.draw_rotated_box(self.drawn_imgs[self.i],d,(0,255,0))
        else:
            if(self.a == 0):
                cv2.rectangle(self.drawn_imgs[self.i],c0,c1,(0,0,255),10)
            else:
                self.draw_rotated_box(self.drawn_imgs[self.i],d,(0,0,255))

    def mouse(self,event,x,y,flags,param):
        if(self.mode == "center"):
            self.center_mouse(event,x,y,flags,param)
        elif(self.mode == "bilateral"):
            self.bilateral_mouse(event,x,y,flags,param)

    def bilateral_mouse(self,event,x,y,flags,param):
        red = (0,0,255)
        blue = (255,0,0)
        if(self.start):
            self.temp_img = self.drawn_imgs[self.i].copy()
            if(self.clicks == 1):
                cv2.line(self.temp_img,(self.b_xy[0]),(x,y),blue,3)
            elif(self.clicks == 2):
                cv2.line(self.temp_img,(self.b_xy[0]),(self.b_xy[1]),blue,3)
                #cv2.circle(self.temp_img,self.xy,5,blue,3) #midpoint
                d = int(np.abs(np.hypot(x - self.xy[0], y - self.xy[1])))
                c0 = (self.xy[0]-d,self.xy[1]-d)
                c1 = (self.xy[0]+d,self.xy[1]+d)
                color = self.check_box(d,c0,c1)
                self.draw_rotated_box(self.temp_img,d,color)

        if event==4: #CLICK UP
            if self.start == False:
                self.start = not self.start
                self.b_xy.append((x,y))
                self.clicks = 1
                # print('set x,y: ', self.b_xy[0])
            elif(self.clicks == 1):
                self.b_xy.append((x,y))
                # print('set x,y: ', self.b_xy[1])
                self.clicks = 2
                dx = self.b_xy[1][0] - self.b_xy[0][0]
                dy = self.b_xy[1][1] - self.b_xy[0][1]

                # note: this assumes your first click is the bottom, the next is the top
                self.a = (np.arctan2(dy,dx)* 180. / np.pi) + 90. # why add 90?
                self.xy = ( int((self.b_xy[0][0] + self.b_xy[1][0])/2) , int((self.b_xy[0][1] + self.b_xy[1][1])/2) )
                # print(np.arctan2(dy,dx) * 180. / np.pi)
                # print(self.xy)
            else:
                # print('lets make a box')
                self.make_crop(d)
                self.reset()

    def center_mouse(self,event,x,y,flags,param):
        #print(event)
        if(self.start):
            self.temp_img = self.drawn_imgs[self.i].copy()
            d = int(np.abs(np.hypot(x - self.xy[0], y - self.xy[1])))
            c0 = (self.xy[0]-d,self.xy[1]-d)
            c1 = (self.xy[0]+d,self.xy[1]+d)
            # print(min(min(c0,c1)))

            color = self.check_box(d,c0,c1)
            cv2.rectangle(self.temp_img,c0,c1,color,6)
            # cv2.imshow('image',tmp)

        if event==4: #CLICK UP
            if self.start == False:
                self.start = not self.start
                self.xy = (x,y)
                # print('set x,y: ', self.xy)
            else:
                # print(self.xy)
                # print(x,y)
                d = int(np.abs(np.hypot(x - self.xy[0], y - self.xy[1])))
                if(d < int(args.min_size/2)): d = int(args.min_size/2)
                self.make_crop(d)


                self.reset()


def interactive(imgs,fs,mode):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)

    c = Context(imgs,fs,mode)

    cv2.imshow('image',c.drawn_imgs[c.i])
    cv2.resizeWindow('image', 1200,800)
    cv2.moveWindow('image', 0,0)
    cv2.setMouseCallback('image', c.mouse)

    print(args.padding)
    if (args.padding > 0):
        c.pad_images(args.padding)

    if (args.outpaint > 0):
        c.outpaint(args.outpaint)

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
            # print(c.i)
            print('next image: ', c.fs[c.i])
            c.reset_xy()
            c.temp_img = c.drawn_imgs[c.i]

            if(c.i >= len(imgs)):
                cv2.destroyWindow('image')
        elif(args.choose and (k==110 or k==121)): #n or y key (only when choose mode is on)
            if (k==121): # y key - accept - copy to yes folder
                choose_save_dest = os.path.join(args.output_folder, "yes")
            else:
                choose_save_dest = os.path.join(args.output_folder, "no")
            print(os.path.abspath(choose_save_dest))
            saveImage(c.drawn_imgs[c.i], os.path.abspath(choose_save_dest), c.fs[c.i])
            c.i+=1
            print('next image: ', c.fs[c.i])
            c.reset_xy()
            c.temp_img = c.drawn_imgs[c.i]
        elif(k==123 or k==108): # l key
            c.i-=1
            if(c.i < 0): c.i = 0
            c.reset_xy()
            c.temp_img = c.drawn_imgs[c.i]
            print('prev image: ', fs[c.i])
        elif(k == 109):
            c.switch_mode()
        else:
            print('pressed: ', k)
def main():
    global args
    args = parse_args()

    os.environ['OPENCV_IO_ENABLE_JASPER']= "true"
    inter = cv2.INTER_CUBIC

    if args.choose:
        yes_path = os.path.abspath(os.path.join(args.output_folder, "yes"))
        no_path = os.path.abspath(os.path.join(args.output_folder, "no"))
        if not os.path.exists(yes_path):
            os.makedirs(yes_path)
        if not os.path.exists(no_path):
            os.makedirs(no_path)

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

    to_load = []
    for root, subdirs, files in os.walk(args.input_folder):
        print('--\nroot = ' + root)

        for subdir in subdirs:
            print('\t- subdirectory ' + subdir)

        for filename in files:
            file_path = os.path.join(root, filename)
            to_load.append(file_path)

    loaded_images = load_images_multi_thread(to_load, args.j__jobs, args.verbose)
    assert len(loaded_images) == len(to_load)
    for i in range(len(loaded_images)):
        if hasattr(loaded_images, 'copy'):
            fs.append(to_load[i])
            imgs.append(loaded_images[i])
    assert len(fs) == len(imgs)

    interactive(imgs,fs,args.mode)


if __name__ == "__main__":
    main()