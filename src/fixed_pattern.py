import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2
import time

################################
#### IMPLEMENTATION CHOICES ####
################################
# display 512 x 512 on projector
# crop and downsample captured image to 512x512

def generate_fixed_pattern(data_dir):
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    io.imsave(data_dir + "zero_im.jpg", np.zeros((512, 512)))
    io.imsave(data_dir + "flooded.jpg", np.ones((512, 512)))
    im_num = 1
    res = []
    for i in range(8):
        for j in range(8):
            im = np.zeros((512, 512))
            im[i::8, j::8] = np.ones((64, 64))
            io.imsave(data_dir + str(im_num).zfill(6) + ".jpg", im)
            res.append(im)
            im_num += 1
    return np.array(res)

    
def process_im(im):
    top, left = (135, 1100)
    bottom, right = (2183, 3148)
    im = im[top:bottom:4, left:right:4]
    return im

'''
ARGUMENTS:
projector - (p*q, k)
camera - (k, m*n)

RETURN:
light transport matrix (m*n, p*q)
'''
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
    
'''
T - (k, p*q)
im_flat - (k, 1)
'''
def get_virtual(T, virtual_light):
    return T.T @ virtual_light

'''
T - (k, p*q)
light - (p*q, 1)
'''
def relight(T, light):
    return T @ light

## Change names of image files ##
'''j = 1
for i in range(8455, 8519):
    fname = "../data/fixed_pattern_cap/IMG_" + str(i) + ".JPG"
    new_name = "../data/fixed_pattern_cap/" + str(j).zfill(6) + ".jpg"
    im = io.imread(fname)
    os.remove(fname)
    io.imsave(new_name, im)
    j += 1'''

## Build projector and camera matrices ##xw
'''projector = []
camera = []
proj_dir = "../data/fixed_pattern/"
cap_dir =  "../data/fixed_pattern_cap/"
for i in range(1, 65):
    pname = proj_dir + str(i).zfill(6) + ".jpg"
    projector.append(io.imread(pname).flatten())

    cname = cap_dir + str(i).zfill(6) + ".jpg"
    cam_im = cv2.cvtColor(io.imread(cname), cv2.COLOR_RGB2GRAY)
    camera.append(process_im(cam_im).flatten())

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

## calculate contributions to virtual image ##
'''cam = np.ones((512, 512))
cam[256:, 256:] = np.zeros((256, 256))
cam = cam.flatten()

dir_name = "../data/fixed_pattern_T/"
save_dir = "../data/fixed_pattern_recon/"
if (not os.path.exists(save_dir)):
    os.mkdir(save_dir)

for i in range(16):
    T = np.load(dir_name + "T" + str(i).zfill(2) + ".npy")
    for j in range(0, 1024*16, 1024):
        np.save(save_dir + str(i).zfill(2) + str(j).zfill(5) + ".npy", \
                get_virtual(T[j:j+1024], cam[i*(1024*16) + j: i*(1024*16) + j + 1024]))
        print("Saved", i, j)
    del T'''

## reconstruct image ##
'''
dir_name = "../data/fixed_pattern_recon/
im = np.zeros(512*512)
for i in range(16):
    for j in range(0, 16*1024, 1024):
        im += np.load(dir_name + str(i).zfill(2) + str(j).zfill(5) + ".npy")
im = im.reshape(512, 512)
plt.imshow(im, cmap="gray")
plt.show()'''

## relight image ##
dir_name = "../data/fixed_pattern_T/"
save_dir = "../data/relight_blue_checker/"

if (not os.path.exists(save_dir)):
    os.mkdir(save_dir)

# build checkerboard
light = np.zeros((512, 512))
for i in range(4):
    for j in range(4):
        if ((i + j) % 2 == 0):
            light[i*(512//4):(i+1)*(512//4), j*(512//4):(j+1)*(512//4)] = \
                np.ones((512//4, 512//4))
                
light = light.flatten()
'''
for i in range(16):
    T = np.load(dir_name + "T" + str(i).zfill(2) + ".npy")
    for j in range(0, 1024*16, 1024):
        np.save(save_dir + str(i).zfill(2) + str(j).zfill(5) + ".npy", \
                relight(T, light))
        print("Saved", i, j)
    del T'''

## add up contributions of relit image ##
relit = np.zeros(512*512)
data_dir = "../data/relight_blue_checker/"
for i in range(16):
    im = np.zeros(1024*16)
    for j in range(0, 1024*16, 1024):
        im += np.load(data_dir + str(i).zfill(2) + str(j).zfill(5) + ".npy").flatten()
    relit[i*(1024*16):(i+1)*(1024*16)] = im
relit = relit.reshape(512, 512)
relit = (relit - np.min(relit))*255/(np.max(relit) - np.min(relit))
io.imsave("../data/relight_blue_checker/relit_checker.jpg", relit)
plt.imshow(relit, cmap="gray")
plt.show()


