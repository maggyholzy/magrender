

import time
import numpy as np
import base64
import argparse
import cv2

background_value = 123

window = [500,500] #rows, columns
window_image = np.full([window[0],window[1],4], 255, dtype=np.uint8) #rgb + alpha, max = 255
window_image[:,:,0:3] = np.full([window[0],window[1],3], background_value, dtype=np.uint8)

title = "MagRender"

cv2.namedWindow(title,cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(title,window[0],window[1])
print(cv2.getWindowImageRect(title))
cv2.moveWindow(title,200,200)



frametime = 0.025

timestep = 0.02
t = 0.0

#points are (3,) shaped vectors


class Drawable:


    def __init__(self, origin_, visible = True, color_ = [1,1,1,1], function = None):
        self.origin = origin_
        self.visible = visible
        self.color = color_
        self.function = function
        pass

    def on_draw(self, cam_):
        #project the drawable object onto the cam
        pass

    
    def frame(self,t):
        if self.function is not None:
            self.function(t, self)

class Cam:
    image = None
    burn_in = False
    D = 0
    n = np.zeros((3,),dtype=float)
    #a rectangular shape on a plane
    def __init__(self, cnl, params_, window_params_, burn_in_ = True):
        self.cnl = cnl #normal vector, array of two points, second point is origin of plane coordinates
        self.n = self.cnl[1,:] - self.cnl[0,:] #N2 - N1
        self.n = self.n / np.linalg.norm(self.n) #normalized normal vector
        self.params = params_ #(4,) w,h,cx,cy
        self.window_params = window_params_ #(4,) w,h,cx,cy
        self.x1 = np.cross(np.asarray([0,0,1]), self.n) #global vector for x of plane-coordinates (2d)
        self.x1 = self.x1 / np.linalg.norm(self.x1) #normalize
        self.y1 = np.linalg.cross(self.n,self.x1)
        self.burn_in = burn_in_

        self.image = np.zeros((self.window_params[1],self.window_params[0],4), dtype = np.uint8)*255 #w,h, 4 channels for rgb including alpha
        self.image[:,:,3] = np.ones((self.window_params[1],self.window_params[0]), dtype= np.uint8)*255 #set alpha high
        
        self.D = np.dot(self.cnl[1], self.n) #sum for plane equation

    def reset_image(self, force = False):
        if (not self.burn_in):
            self.image = np.zeros((self.window_params[1],self.window_params[0],4), dtype = np.uint8)*255 #w,h, 4 channels for rgb including alpha
            self.image[:,:,3] = np.ones((self.window_params[1],self.window_params[0]), dtype= np.uint8)*255 #set alpha high

    def get_plane_coords(self, point_): 
        
        n = self.cnl[1,:] - self.cnl[0,:] #N2 - N1
        # n = n / np.linalg.norm(self.n) #normalize n, already normalized
        a = -(np.dot(point_, self.n)) + self.cnl[1,:]
        alpha = point_ + a * self.n - self.cnl[1,:] #vector lies on plane, still 3D

        x = np.dot(alpha,self.x1)
        y = np.dot(alpha,self.y1)

        return np.squeeze(np.asarray([y,x]))

    def draw_sprite(self,sprite:np.ndarray,coordinate):
        sh = sprite.shape
        # print(f"sprite.shape:{sprite.shape}, self.image.shape:{self.image.shape}")
        # print(f"coordinate:{coordinate}")
        ratios = self.window_params[0:2] / self.params[0:2] #ratio of widths and heights
        # print(f"ratios:{ratios}, self.window_params[2:]:{self.window_params[2:]}")
        img_coord = np.add(coordinate - self.params[2:]*ratios, self.window_params[2:]) #coordinate of center of sprite in pixel coordinates
        # print(f"img_coord:{img_coord}")
        img_coord_int = np.array([int(i) for i in img_coord])
        # print(f"img_coord_int:{img_coord_int}")
        # print(f"{img_coord_int[0]-sh[0]//2},{img_coord_int[0]+sh[0]//2+1}")
        # print(f"{img_coord_int[1]-sh[1]//2},{img_coord_int[1]+sh[1]//2+1}")
        # print(sprite)
        self.image[img_coord_int[0]-sh[0]//2:img_coord_int[0]+sh[0]//2+1:, 
                   img_coord_int[1]-sh[1]//2:img_coord_int[1]+sh[1]//2+1:,
                   :] = sprite
        pass

    def get_image(self):
        return self.image
    
    def set_burnin(self, burn_in_):
        self.burn_in = burn_in_
    def get_burnin(self):
        return self.burn_in

    def __str__(self):
        return f'parameters: {self.params}' + '\n' + f'window parameters: {self.window_params}'+ '\n' + f'normal line: {self.cnl}'



class Point (Drawable):


    def __init__(self, origin_, point_, function = None, visible=True, size=11):
        self.point = point_ #(3,)
        self.function = function
        self.size = size
        self.sprite()
        self.color = [1,1,1]*255
        super().__init__(origin_, visible, function = function)

    def on_draw(self, cam_:Cam):
        try:
            a = cam_.get_plane_coords(self.point)
            # print(a)
            cam_.draw_sprite(self.sprite,a)


        except Exception as e:
            print(f"exception:{e}")

        pass

    def sprite(self): #size should be an odd number so image has center
        self.sprite = np.ones((self.size,self.size,4), dtype = np.uint8)*255 #size x size white square, can make dot later

    def set_point(self,point_):
        self.point = point_

    def set_color(self,color_):
        self.color = color_
    
    def frame(self,t):
        try:
            self.function(t, self)
        except Exception as e:
            print(f"error: {e}")
            print(f"{self.function}")
        
def helix(t, p:Point):
    point = np.asarray([0,0,0],dtype=np.float32)
    r = np.abs(4 + 4 * (np.sin( t * 0.24))/(np.cos( t * 0.24)+0.20))
    point[0] = np.cos(t*15) * r
    point[1] = np.sin(t*15) * r
    point[2] = -290 + 12 * t

    point.resize((1,3))

    print(point)

    p.set_point(point)
    return point

    
def frame():
    global cams, drawables, t, timestep, title, window_image

    window_image = np.full([window[0],window[1],4], 255, dtype=np.uint8) #rgb + alpha, max = 255
    window_image[:,:,0:3] = np.full([window[0],window[1],3], background_value, dtype=np.uint8)

    print("frame")
    
    cam: Cam
    for cam in cams:
        if(not cam.get_burnin()): cam.reset_image()
        obj: Drawable
        for obj in drawables:
            obj.frame(t)
            obj.on_draw(cam_=cam)
            pass
        #generate image for each cam
        cam_image = cam.get_image().copy()
        window_image[window[0]//2 - cam_image.shape[0]//2:window[0]//2 + cam_image.shape[0]//2,
                     window[1]//2 - cam_image.shape[1]//2:window[1]//2 + cam_image.shape[1]//2,
                     :] = cam_image.copy()

    cv2.imshow(title,window_image[:,:,0:3])

    t += timestep

    return



#initiate cam
length1 = 5
cam1 = Cam(cnl=np.reshape([0,0,0,1,1,1],(2,3))*length1, 
            params_=np.array([length1,length1,0,0]), #(4,) w,h,cx,cy, center is at origin of N2
            window_params_=np.array([(window[0]-100//2),(window[1]-100//2),(window[0]-100//2)/2,(window[1]-100//2)/2], dtype=np.uint16)) #(4,) w,h,cx,cy

cams = []
cams.append(cam1)
for cam in cams:
    print(cam)

#initiate drawables

drawables = []

helix_t = Point(np.reshape([0,0,0],shape=(1,3)),np.reshape([0,0,0],shape=(1,3)), function=helix, size=1)
drawables.append(helix_t)

while True:
    frame()
    cv2.waitKey((int)(timestep*1000 / 20))

cv2.destroyAllWindows()