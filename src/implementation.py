import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2

def generate_fixed_pattern(data_dir):
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    io.imsave(data_dir + "zero_im.jpg", np.zeros((512, 512)))
    io.imsave(data_dir + "flooded.jpg", np.ones((512, 512)))
    im_num = 1
    res = []
    for i in range(8):
        for j in range(8):
            im = np.zeros((256, 256))
            im[i::8, j::8] = np.ones((32, 32))
            io.imsave(data_dir + str(im_num).zfill(6) + ".jpg", im)
            res.append(im)
            im_num += 1
    return np.array(res)
    
def generate_bernoulli(data_dir):
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    res = []
    for i in range(100):
        im = np.random.randint(2, size=(256, 256))
        io.imsave(data_dir + str(i+1).zfill(6) + ".jpg", im)
        res.append(im)
    return np.array(res)

def generate_bar_scan(data_dir):
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    res = []
    for i in range(128):
        im = np.zeros((256, 256))
        im[i*2:(i+1)*2, :] = np.ones((2, 256))
        io.imsave(data_dir + str(i+1).zfill(6) + ".jpg", im)
        res.append(im)
    return np.array(res)

def calculate_ltm(projector, camera):
    pass
    
def get_virtual(T, virtual_light):
    pass

def relight(T, light):
    pass

def calculate_ltm(projector, camera):
    pass