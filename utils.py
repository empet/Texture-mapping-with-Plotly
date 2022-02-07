import numpy as np
from numpy import pi, sin, cos

scene_style = dict(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   scene_aspectmode="data")

def surface(rows, cols):
    x = np.linspace(-pi, pi, cols) 
    y = np.linspace(-pi, pi, rows)
    x, y = np.meshgrid(x,y)
    z = 0.5*cos(x/2) + 0.2*sin(y/4)
    return x, y, z

def sphere(rows, cols):
    u, v = np.meshgrid(np.linspace(0, 2*pi, cols), np.linspace(-pi/2, pi/2, rows))
    return cos(u)*cos(v), sin(u)*cos(v), sin(v)

def cylinder(rows, cols, R=1):
    u, v = np.meshgrid(np.linspace(0, 2*pi, cols), np.linspace(0, 2, rows))
    return R*cos(u), R*sin(u), v

def cone(rows, cols):
    #NOTICE that here v, which has a longest range [0, 2 pi], is put on the first position to ensure that
    # v, u have more columns than rows, similar to the image, img
    v, u = np.meshgrid( np.linspace(0, 2*pi, cols), np.linspace(0, 1, rows))
    return u*cos(v), u*sin(v), 1-u
