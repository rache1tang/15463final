import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2

################################
#### IMPLEMENTATION CHOICES ####
################################
# display 512 x 512 on projector
# crop and downsample captured image to 512x512

FIXED_PATTERN = 0
ADAPTIVE = 1
ROW_SCAN = 2

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
    
def subdivide(pattern, captured, frame_order):
    pass
    
def process_im(im, data_id):
    if (data_id == FIXED_PATTERN):
        threshold = 150
        top, left = (135, 1100)
        bottom, right = (2183, 3148)
    im = im[top:bottom:4, left:right:4]
    return im

io.imsave("../data/fixed_pattern/zero_im.jpg", np.zeros((256, 256)))
io.imsave("../data/fixed_pattern/flooded.jpg", np.ones((256, 256)))


def calculate_ltm(projector, camera):
    # camera = T @ projector
    '''
    ci = P^T @ ti
    -------------
    ci - (k, 1)
    P^T - (k, p*q)
    ti - (p*q, 1)
    -------------
    i in range(m*n)
    '''

    pq, _ = projector.shape
    _, mn = camera.shape
    PT = projector.T
    T = np.zeros((mn, pq))
    for i in range(mn//1024):
        t, _, _, _ = np.linalg.lstsq(PT, camera[:, i*1024:(i+1)*1024], rcond=None)
        T[i*1024:(i+1)*1024, :] = t.T
        print("finished", (i+1)*1024, "/", mn)
    return T
    

## Change names of image files ##
'''j = 1
for i in range(8455, 8519):
    fname = "../data/fixed_pattern_cap/IMG_" + str(i) + ".JPG"
    new_name = "../data/fixed_pattern_cap/" + str(j).zfill(6) + ".jpg"
    im = io.imread(fname)
    os.remove(fname)
    io.imsave(new_name, im)
    j += 1'''

## Build projector and camera matrices ##
'''projector = []
camera = []
proj_dir = "../data/fixed_pattern/"
cap_dir =  "../data/fixed_pattern_cap/"
for i in range(1, 65):
    pname = proj_dir + str(i).zfill(6) + ".jpg"
    projector.append(io.imread(pname).flatten())

    cname = cap_dir + str(i).zfill(6) + ".jpg"
    cam_im = cv2.cvtColor(io.imread(cname), cv2.COLOR_RGB2GRAY)
    camera.append(process_im(cam_im, FIXED_PATTERN).flatten())
projector = np.array(projector).T
camera = np.array(camera)'''


## Calculate T matrix in segments ##
'''dir_name = "../data/fixed_pattern_T/"
if (not os.path.exists(dir_name)):
    os.mkdir(dir_name)
for i in range(camera.shape[1]//(1024*16)):
    T = calculate_ltm(projector, camera[:, i*(1024*16):(i+1)*(1024*16)])
    fname = dir_name + "T" + str(i).zfill(2) + ".npy"
    np.save(fname, T)
    del T'''
