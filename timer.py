import os
import numpy as np
import cv2
import pyttsx3
from datetime import datetime, timedelta
import time
from playsound import playsound
from threading import Thread

# configuration
race = '100m_Hurdles'
#race = '110m_Hurdles'
#race = '100m'
#race = '1600m'
#race = '4x100m'
#race = '400m'
#race = '300m_Hurdles'
#race = '800m'
#race = '200m'
#race = '3200m'
#race = '4x400m'

start_commands=False
record_video=True
video_dir = 'C:/Users/johnh/Desktop/videos/'  # './'
video_type = 'mp4' # avi, mp4
meet='Timpview Meet'
camera1 = 0 # 0=front, 1=rear, 2=external
camera2 = None
logo = 'timpview.png'
# 1920 x 1080
# 1280 x 720
screen_res = 1280, 720
#screen_res = 1920, 1080

class ThreadedCamera(object):
    def __init__(self, source=0):
        # video capture
        self.cap1 = cv2.VideoCapture(source,cv2.CAP_DSHOW)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, screen_res[0])
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_res[1])

        self.thread = Thread(target=self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def update(self):
        while True:
            time.sleep(0.01)
            if self.cap1.isOpened():
                (self.status, self.frame) = self.cap1.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None

    def size(self):
        w = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))    
        h = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w,h

    def release(self):
        self.cap1.release()
        return None

class ThreadedWriter(object):
    def __init__(self,vtype,w,h):

        self.width = w
        self.height = h

        if vtype=='avi':
            # Write video.avi
            self.out = cv2.VideoWriter(video_dir+dname+'.'+video_type, \
                                       cv2.VideoWriter_fourcc(*'XVID'), \
                                       30, (w,h))
        else:
            # Write video.mp4
            self.out = cv2.VideoWriter(video_dir+dname+'.'+video_type, \
                                       cv2.VideoWriter_fourcc(*'MP4V'), \
                                       30, (w,h))
        
        self.thread = Thread(target=self.write, args =())
        self.thread.daemon = True
        self.thread.start()

        # queue of frames to write
        self.frames = []
        self.n = 0
        self.block = False

    def write(self):
        while True:
            time.sleep(0.001)
            if self.n>=1:
                # write frame
                self.out.write(self.frames[0])
                # remove frame from list
                #self.block = True # block access to frames, n
                self.frames.pop(0)
                self.n -= 1
                #self.block = False

    def add_frame(self,frame):
        if frame is not None:
            self.frames.append(frame)
            self.n += 1
        return

    def release(self):
        self.out.release()
        return None

class ThreadedStarter(object):
    def __init__(self,commands=False,sprint=True):
        if commands:
            self.st = None
            self.thread = Thread(target=self.start, args =(sprint,))
            self.thread.start()
        else:
            self.st = datetime.now()

    def start(self,sprint):
        # Start the Race with Audio Commands
        if sprint:
            engine.say('On your marks')
            engine.runAndWait()
            time.sleep(0.1)
        # Set only for distance events
        engine.say('Set')
        engine.runAndWait()
        # gun sound
        playsound('start.mp3')
        # subtract length of audio file
        self.st = datetime.now() - timedelta(seconds=0.6531)
        return

# read logo
img = cv2.imread(logo,cv2.IMREAD_UNCHANGED)
# resize logo
width=img.shape[1]; height=img.shape[0]
hlogo=50; scale_logo=hlogo/height
wlogo=int(width*scale_logo)
img = cv2.resize(img, (wlogo,hlogo),\
                 interpolation = cv2.INTER_AREA)
alpha = img[:,:,3] / 255.0
beta = []
for i in range(3):
    beta.append(img[:,:,i] * (alpha))

# read roadrunner logo
img = cv2.imread('rr.png',cv2.IMREAD_UNCHANGED)
# resize logo
width=img.shape[1]; height=img.shape[0]
hrr=50; scale_rr=hrr/height
wrr=int(width*scale_rr)
img = cv2.resize(img, (wrr,hrr),\
                 interpolation = cv2.INTER_AREA)
arr = img[:,:,3] / 255.0
brr = []
for i in range(3):
    brr.append(img[:,:,i] * (arr))

# read award icons, 1st-3rd
award1 = cv2.imread('Award1.png',cv2.IMREAD_UNCHANGED)
award2 = cv2.imread('Award2.png',cv2.IMREAD_UNCHANGED)
award3 = cv2.imread('Award3.png',cv2.IMREAD_UNCHANGED)
# resize logo
width=award1.shape[1]; height=award1.shape[0]
haward=300; scale_award=haward/height
waward=int(width*scale_award)
award1 = cv2.resize(award1, (waward,haward),\
                    interpolation = cv2.INTER_AREA)
award2 = cv2.resize(award2, (waward,haward),\
                    interpolation = cv2.INTER_AREA)
award3 = cv2.resize(award3, (waward,haward),\
                    interpolation = cv2.INTER_AREA)
alpha1 = award1[:,:,3] / 255.0
alpha2 = award2[:,:,3] / 255.0
alpha3 = award3[:,:,3] / 255.0
beta1=[]; beta2=[]; beta3=[]
for i in range(3):
    beta1.append(award1[:,:,i] * (alpha1))
    beta2.append(award2[:,:,i] * (alpha2))
    beta3.append(award3[:,:,i] * (alpha3))

date=datetime.now().strftime("%B %d, %Y")
meet = meet + ' ' + date

# directory for results
dname = 'Race_'+datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+'-'+race
try:
    os.mkdir(dname)
except:
    print('Directory '+dname+' exists')

# start audio engine
engine = pyttsx3.init()

# place: 1st, 2nd, 3rd...
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

# start video capture
c1 = ThreadedCamera(camera1)

# set up video writing
if record_video:
    w,h = c1.size()
    # start video writer
    out1 = ThreadedWriter(video_type,w,h)
    
# capture 1 image for setup
img = None
while img is None:
    img = c1.grab_frame()
# desired screen resolution
resize = False
if resize:
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
else:
    scale = 1.0
# resized window width and height
width=img.shape[1]
height=img.shape[0]
# scaling for thumbnails
sthumb = 0.1
sw = int(width*sthumb)
sh = int(height*sthumb)
dthumb = (sw,sh)
# scaling for window to match desired screen resolution
window_width = int(width * scale)
window_height = int(height * scale)
cv2.namedWindow('Race Timer',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Race Timer', window_width+100,window_height)

# create results file
fid = open('./'+dname+'/'+dname+'.csv','w')
fid.write('Place,Time\n')
fid.close()

started = False
nf=0; finishers=[]; frames=[]
while True:
    img_raw = c1.grab_frame()
    img = img_raw.copy()
    # Put current DateTime on each frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font2 = cv2.FONT_HERSHEY_TRIPLEX
    if started:
        # current time mm.ss.00
        if starter.st is not None:
            diff = (datetime.now()-starter.st).total_seconds()
        else:
            diff = 0.0
        dm,ds = divmod(diff, 60)
        str_time = '{0:02d}:{1:05.2f}'.format(int(dm), ds)
        # large text at the top with race time
        cv2.putText(img,str_time,(320,170), \
                    font, 6.8,(0,0,0),20,cv2.LINE_AA)
        cv2.putText(img,str_time,(320,170), \
                    font, 6.8,(255,255,255),10,cv2.LINE_AA)
        # insert logo with transparency
        img1 = img[665:665+hlogo,350:350+wlogo,:]
        for c in range(0,3):
            color = img1[:,:,c] * (1.0-alpha)
            img1[:,:,c] = color + beta[c]
        img[665:665+hlogo,350:350+wlogo,:] = img1
        # insert roadrunner logo with transparency
        yloc = screen_res[1]-60; xloc = screen_res[0]-150
        img1 = img[yloc:yloc+hrr,xloc:xloc+wrr,:]
        for c in range(0,3):
            color = img1[:,:,c] * (1.0-arr)
            img1[:,:,c] = color + brr[c]
        img[yloc:yloc+hrr,xloc:xloc+wrr,:] = img1        
        # meet text info
        cv2.putText(img,meet,(370+wlogo,705), \
                    font, 1.5,(0,0,0),8,cv2.LINE_AA)
        cv2.putText(img,meet,(370+wlogo,705), \
                    font, 1.5,(255,255,255),4,cv2.LINE_AA)
        # Insert recorded times and thumbnails
        scroll = max(0,nf-10)
        if scroll==0:
            # no scrolling - display top 10
            for i in range(scroll,nf):
                x_offset=10
                y_offset=5+(i-scroll)*70
                s_img = frames[i]
                img[y_offset:y_offset+s_img.shape[0],\
                    x_offset:x_offset+s_img.shape[1]] = s_img
                cv2.putText(img,finishers[i],(93,60+(i-scroll)*70), \
                            font, 1.2,(0,0,0),6,cv2.LINE_AA)
                cv2.putText(img,finishers[i],(93,60+(i-scroll)*70), \
                            font, 1.2,(255,255,255),2,cv2.LINE_AA)
        else:
            # always display top 3
            for i in range(3):
                x_offset=10
                y_offset=5+i*70
                s_img = frames[i]
                img[y_offset:y_offset+s_img.shape[0],\
                    x_offset:x_offset+s_img.shape[1]] = s_img
                cv2.putText(img,finishers[i],(93,60+i*70), \
                            font, 1.2,(0,0,0),6,cv2.LINE_AA)
                cv2.putText(img,finishers[i],(93,60+i*70), \
                            font, 1.2,(255,255,255),2,cv2.LINE_AA)
            # display the rest
            for i in range(scroll+4,nf):
                x_offset=10
                y_offset=5+(i-scroll)*70
                s_img = frames[i]
                img[y_offset:y_offset+s_img.shape[0],\
                    x_offset:x_offset+s_img.shape[1]] = s_img
                cv2.putText(img,finishers[i],(93,60+(i-scroll)*70), \
                            font, 1.2,(0,0,0),6,cv2.LINE_AA)
                cv2.putText(img,finishers[i],(93,60+(i-scroll)*70), \
                            font, 1.2,(255,255,255),2,cv2.LINE_AA)
                
    # Display the image
    cv2.imshow('Race Timer',img)
    if record_video:
        out1.add_frame(img)
    # wait for keypress
    k = cv2.waitKey(10)
    if k==32:        
        if not started:
            started = True
            starter = ThreadedStarter(start_commands)
        else:
            nf += 1
            finishers.append(str(nf)+' '+str_time)
            # insert awards for 1st, 2nd, 3rd
            if nf==1:
                img1 = img[365:365+haward,0:waward,:]
                for c in range(0,3):
                    color = img1[:,:,c] * (1.0-alpha1)
                    img1[:,:,c] = color + beta1[c]
                img[365:365+haward,0:waward,:] = img1
            if nf==2:
                img1 = img[365:365+haward,0:waward,:]
                for c in range(0,3):
                    color = img1[:,:,c] * (1.0-alpha2)
                    img1[:,:,c] = color + beta2[c]
                img[365:365+haward,0:waward,:] = img1
            if nf==3:
                img1 = img[365:365+haward,0:waward,:]
                for c in range(0,3):
                    color = img1[:,:,c] * (1.0-alpha3)
                    img1[:,:,c] = color + beta3[c]
                img[365:365+haward,0:waward,:] = img1
            # 1st place, 2nd place, etc.
            cv2.putText(img,ordinal(nf)+' Place',(330,650), \
                        font, 5,(0,0,0),20,cv2.LINE_AA)
            cv2.putText(img,ordinal(nf)+' Place',(330,650), \
                        font, 5,(255,255,255),8,cv2.LINE_AA)
            resized = cv2.resize(img_raw, dthumb,\
                                 interpolation = cv2.INTER_AREA)
            frames.append(resized)
            # write image to directory
            cv2.imwrite('./'+dname+'/Finisher_{0:02d}.png'.format(nf),img)
            # record results in CSV file
            fid = open('./'+dname+'/'+dname+'.csv','a')
            fid.write("{0:d},'{1:s}\n".format(nf,str_time))
            fid.close()
            # add multiple frames of the finisher
            if record_video:
                for i in range(5):
                    out1.add_frame(img)
    elif k==27 or k==ord('q'):
        # esc or 'q' to quit
        break
    elif k==ord('f'):
        playsound('start.mp3')
        playsound('start.mp3')
        engine.say('False Start, Return to the Starting Line')
        engine.runAndWait()
    elif k==ord('r'):
        # restart
        started = False
        nf=0; finishers=[]; frames=[]
        out1.release()
        dname = 'Race_'+datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        if record_video:
            w,h = c1.size()
            # start video writer
            out1 = ThreadedWriter(video_type,w,h)
        fid = open('./'+dname+'/'+dname+'.csv','w')
        fid.write('Place,Time\n')
        fid.close()
        
c1.release()
cv2.destroyAllWindows()
if record_video:
    # finish writing frames up to 10 sec afterwards
    ft = time.time()
    if out1.n>=1:
        print('Waiting to write ', str(out1.n), ' frames to video')
    while out1.n>=1 and (time.time()-ft)<10.0:
        time.sleep(0.01)
    out1.release()

