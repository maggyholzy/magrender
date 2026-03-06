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

def helix(t, p:mg.Point):
    point = np.asarray([0,0,0],dtype=np.float32)
    r = np.min([np.abs(2 + 4 * np.abs((np.tan( t * 0.24)))),200])
    point[0] = np.cos(t*20) * r
    point[1] = np.sin(t*20) * r
    point[2] = -450 + 16 * t

    point.resize((1,3))

    print(point)

    color = np.asarray([np.cos(t)*0.50+0.50,np.cos(t + np.pi*2/3)*0.50+0.50, np.cos(t + np.pi*4/3)*0.50+0.50],dtype=np.float32)
    color = color * 255.0
    print(f"color:{color}")
    p.set_color(color_=color)
    p.set_sprite()

    p.set_point(point)

    return point

helix_t = mg.Point(np.reshape([0,0,0],shape=(1,3)),np.reshape([0,0,0],shape=(1,3)), function=helix, size=3)
mg.drawables.append(helix_t)

axis_length = 400

x_axis = mg.Line(np.array([0,0,0],dtype=np.float32),np.array([[0,0,0],[axis_length,0,0]],dtype=np.float32),size = 1,d_ = 2.0)
x_axis.set_tag("oneshot")
mg.drawables.append(x_axis)
y_axis = mg.Line(np.array([0,0,0],dtype=np.float32),np.array([[0,0,0],[0,axis_length,0]],dtype=np.float32),size = 1,d_ = 2.0)
y_axis.set_tag("oneshot")
mg.drawables.append(y_axis)
z_axis = mg.Line(np.array([0,0,0],dtype=np.float32),np.array([[0,0,0],[0,0,axis_length]],dtype=np.float32),size = 1,d_ = 2.0)
z_axis.set_tag("oneshot")
mg.drawables.append(z_axis)



while True:
    mg.frame()
    


