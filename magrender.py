import time
import numpy as np
import base64
import argparse
import cv2


background_value = 126

window = [800,800] #rows, columns
window_image = np.full([window[0],window[1],4], 255, dtype=np.uint8) #rgb + alpha, max = 255
window_image[:,:,0:3] = np.full([window[0],window[1],3], background_value, dtype=np.uint8)

title = "MagRender"

cv2.namedWindow(title,cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(title,window[0],window[1])
print(cv2.getWindowImageRect(title))
cv2.moveWindow(title,200,1800)



frametime = 0.025

timestep = 0.02
t = 0.0

#points are (3,) shaped vectors

cams = []
drawables = []


class Drawable:


    def __init__(self, origin_, visible = True, function = None):
        self.origin = origin_
        self.visible = visible
        self.function = function
        self.tags = []
        pass

    def on_draw(self, cam_):
        #project the drawable object onto the cam
        pass

    
    def frame(self,t):
        if self.function is not None:
            self.function(t, self)

    def set_tag(self, tag_ : str):
        self.tags.append(tag_)

    def remove_tag(self, tag_: str):
        try:
            self.tags.remove(tag_)
        except Exception as e:
            print(f"error: {e}")
            print(f"{self.tags}")

            
    def frame(self,t):
        if(self.function is not None):
            try:
                self.function(t, self)
            except Exception as e:
                print(f"error: {e}")
                print(f"line:{e.with_traceback}")
                print(f"{self.function}")

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
        self.ratios = self.window_params[0:2] / self.params[0:2] #ratio of widths and heights
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
        
        img_coord = np.add(coordinate - self.params[2:]*self.ratios, self.window_params[2:]) #coordinate of center of sprite in pixel coordinates
        
        img_coord_int = np.array([int(i) for i in img_coord])
        self.image[img_coord_int[0]-sh[0]//2:img_coord_int[0]+sh[0]//2+1:, 
                   img_coord_int[1]-sh[1]//2:img_coord_int[1]+sh[1]//2+1:,
                   :] = sprite
        pass

    def get_image(self):
        return self.image[::-1,:,:].copy()
    
    def set_burnin(self, burn_in_):
        self.burn_in = burn_in_
    def get_burnin(self):
        return self.burn_in

    def __str__(self):
        return f'parameters: {self.params}' + '\n' + f'window parameters: {self.window_params}'+ '\n' + f'normal line: {self.cnl}'



class Point (Drawable):


    def __init__(self, origin_, point_, function = None, visible=True, size=11,color_ = [255,255,255]):
        self.point = point_ #(3,)
        self.function = function
        self.size = size
        self.color = color_
        self.set_sprite()
        super().__init__(origin_, visible, function = function)

    def on_draw(self, cam_:Cam):
        try:
            a = cam_.get_plane_coords(self.point)
            # print(a)
            cam_.draw_sprite(self.sprite,a)
        except Exception as e:
            print(f"exception:{e}")
        pass

    def set_sprite(self): #size should be an odd number so image has center
        self.sprite : np.ndarray
        self.sprite = np.ones((self.size,self.size,4), dtype = np.uint8)*255 #size x size white square, can make dot later
        for i in range(3):
            self.sprite[:,:,i].fill(self.color[i])
        
    def set_color(self, color_ : np.ndarray):
        #color is an ndarray, (3,), int32 between 0 and 255
        color_ = np.array([int(i) for i in color_])
        color_.clip(0,255)
        self.color = color_
        self.set_sprite()
        # self.sprite(self) #rebuild sprite, not done every frame

    def get_color(self):
        return self.color

    def set_point(self,point_):
        self.point = point_

    

class Line (Drawable):
    def __init__(self, origin_, points_,d_ = 1.0, function = None, visible=True, size=11):
        self.points = points_ #(2,3) row x col
        self.function = function
        self.size = size
        self.sprite()
        self.d = d_ #draw interval, float32
        super().__init__(origin_, visible, function = function)

    def on_draw(self, cam_:Cam):
        
        #interpolate 
        vec = self.points[1,:] - self.points[0,:]
        vec_len = np.linalg.norm(vec)
        vec = vec / vec_len #normalize
        point = np.copy(self.points[0,:])
        a1 = cam_.get_plane_coords(self.points[0,:])
        a2 = cam_.get_plane_coords(self.points[1,:])
        vec2 = (a2-a1)
        a_len = np.linalg.norm(vec2)
        vec2 = vec2 / a_len #normalize

        a = a1.copy()

        # while (np.linalg.norm(point - self.points[0,:]) < vec_len):

        #     try:
        #         a = cam_.get_plane_coords(point)
        #         # print(a)
        #         cam_.draw_sprite(self.sprite,a)
                

        #     except Exception as e:
        #         print(f"exception:{e}")
        #     point += vec * self.d

        # a = cam_.get_plane_coords(self.points[1,:])
        # # print(a)
        # cam_.draw_sprite(self.sprite,a)

        # pass
        while (np.linalg.norm(a - a1) < a_len):
            try:
                cam_.draw_sprite(self.sprite,a)
                a += vec2 * self.d
            except Exception as e:print(f"exception:{e}")

        pass

    def set_points(self,points_):
        self.points = points_ #2,3 float32

    def sprite(self): #size should be an odd number so image has center
        self.sprite = np.ones((self.size,self.size,4), dtype = np.uint8)*255 #size x size white square, can make dot laterv


        


    
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
            if(obj.tags.__contains__("oneshot") and not obj.tags.__contains__("drawn")):
                obj.frame(t)
                obj.on_draw(cam_=cam)
                obj.set_tag("drawn")
            elif(obj.tags.__contains__("oneshot") and obj.tags.__contains__("drawn")):
                obj.frame(t)
                pass
            else:
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
    cv2.waitKey((int)(1))

    return

for cam in cams:
    print(cam)


cv2.moveWindow(title,200,1800)


cv2.destroyAllWindows()