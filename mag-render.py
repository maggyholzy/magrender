

import time
import numpy as np
import base64
import argparse
import cv2

window = [500,500] #rows, columns
window_image = np.full([window[0],window[1],4], fill_value= 255, dtype=np.uint8) #rgb + alpha, max = 255

title = "MagRender"

cv2.namedWindow(title,cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(title,window[0],window[1])
cv2.moveWindow(title,200,200)

frametime = 0.1

timestep = 0.05
t = 0.0

#points are (3,) shaped vectors

class Cam:
    image = None
    burn_in = False
    D = 0
    n = np.zeros((3,),dtype=float)
    #a rectangular shape on a plane
    def __init__(self, norm_line_,params_, window_params_, burn_in_ = False):
        self.norm_line = norm_line_ #normal vector, array of two points, second point is origin of plane coordinates
        self.params = params_ #(4,) w,h,cx,cy
        self.window_params = window_params_ #(4,) w,h,cx,cy
        self.burn_in = burn_in_

        self.image = np.zeros((self.params[1],self.params[0],4), dtype = np.uint8)*255 #w,h, 4 channels for rgb including alpha
        self.image[:,:,3] = np.ones((self.params[1],self.params[0]), dtype= np.uint8)*255 #set alpha high
        
        self.n = self.norm_line[1] - self.norm_line[0] #take norm vector
        self.n = self.n / np.linalg.norm(self.n) #normalizing n
        self.D = np.dot(self.norm_line[1], self.n) #sum for plane equation

    def reset_image(self, force = False):
        if (not self.burn_in):
            self.image = np.zeros((self.params[1],self.params[0],4), dtype = np.uint8)*255 #w,h, 4 channels for rgb including alpha
            self.image[:,:,3] = np.ones((self.params[1],self.params[0]), dtype= np.uint8)*255 #set alpha high

    def get_plane_coords(self, point):
        self.n = self.norm_line[1] - self.norm_line[0] #take norm vector
        self.n = self.n / np.linalg.norm(self.n) #normalizing n
        self.D = np.dot(self.norm_line[1], self.n) #sum for plane equation

        point1 = point + (self.D - np.dot(self.n,point)) * self.n #projected point, lying on plane

    def get_image(self):
        return self.image
    
    def set_burnin(self, burn_in_):
        self.burn_in = burn_in_
    def get_burnin(self):
        return self.burn_in

    def __str__(self):
        return f'parameters: {self.params}' + '\n' + f'window parameters: {self.window_params}'+ '\n' + f'normal line: {self.norm_line}'


class Drawable:


    def __init__(self, origin_, visible = True, color_ = [1,1,1,1]):
        self.origin = origin_
        self.visible = visible
        self.color = color_
        pass

    def on_draw(self, cam_):
        #project the drawable object onto the cam
        pass

class Point (Drawable):

    def on_draw(self, cam_:Cam):
        
        #where does the point project onto the cam?


        pass
    
    def __init__(self, origin_, point_, visible=True):
        self.point = point_ #(3,)
        super().__init__(origin_, visible)

    def sprite(self,size=3): #default 3x3 image for dot
        self.sprite = np.ones((size,size,4), dtype = np.uint8)*255 #size x size white square, can make dot later
        
        
    
def frame():
    global cams, drawables, t, timestep

    
    cam: Cam
    for cam in cams:
        cam.reset_image()
        obj: Drawable
        for obj in drawables:
            obj.on_draw(cam_=cam)
            pass
        #generate image for each cam
        cam_image = cam.get_image()


    t += timestep

    return

#initiate cam
cam1 = Cam(norm_line_=np.reshape([0,0,0,1,1,1],(2,3))*100.0, 
            params_=np.array([100,100,0,0]), 
            window_params_=np.array([(window[0]-100/2),(window[1]-100/2),100,100]))
cams = []
cams.append(cam1)
for cam in cams:
    print(cam)

#initiate drawables

helix_t = Point(np.reshape([0,0,0],shape=(1,3)),np.reshape([0,0,0],shape=(1,3)))


drawables = []

while True:
    frame()
    time.sleep(frametime)

