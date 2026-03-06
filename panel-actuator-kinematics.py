import magrender as mg
import numpy as np
import base64
import argparse
import cv2


n = 1000 #number of columns

t_1 = 10 #finish time, seconds

#initiate cam

length1 = 5
cam1 = mg.Cam(cnl=np.reshape([0,0,0,1,1,1],(2,3))*length1, 
            params_=np.array([length1,length1,0,0]), #(4,) w,h,cx,cy, center is at origin of N2
            window_params_=np.array([(mg.window[0]-100//2),(mg.window[1]-100//2),(mg.window[0]-100//2)/2,(mg.window[1]-100//2)/2], dtype=np.uint16)) #(4,) w,h,cx,cy
mg.cams.append(cam1)

#initiate drawables

vartab = {}
vartab["ind"] = []
vartab["t"] = []
vartab["r1"] = []
vartab["r0"] = []
vartab["images"] = []


x_axis = mg.Line(np.array([0,0,0],dtype=np.float32),np.array([[0,0,0],[200,0,0]],dtype=np.float32),size = 1,d_ = 0.50)
mg.drawables.append(x_axis)
y_axis = mg.Line(np.array([0,0,0],dtype=np.float32),np.array([[0,0,0],[0,200,0]],dtype=np.float32),size = 1,d_ = 2)
mg.drawables.append(y_axis)
z_axis = mg.Line(np.array([0,0,0],dtype=np.float32),np.array([[0,0,0],[0,0,200]],dtype=np.float32),size = 1,d_ = 5)
mg.drawables.append(z_axis)



while True:
    mg.frame()
    


