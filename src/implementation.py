import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import OrthogonalMatchingPursuit, orthogonal_mp
import time

GRAYSCALE = 0
RED = 1
GREEN = 2
BLUE = 3

OMP = "OMP"
ROMP = "ROMP"
LEAST_SQUARES = "lstsq"

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
            io.imsave(data_dir + str(im_num).zfill(6) + ".png", im)
            res.append(im)
            im_num += 1
    return np.array(res)
    
def generate_bernoulli(data_dir):
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    res = []
    for i in range(512):
        im = np.random.randint(2, size=(256, 256))
        io.imsave(data_dir + str(i+1).zfill(6) + ".png", im)
        res.append(im)
    return np.array(res)

def generate_bar_scan(data_dir):
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    res = []
    for i in range(128):
        im = np.zeros((256, 256))
        im[i*2:(i+1)*2, :] = np.ones((2, 256))
        io.imsave(data_dir + str(i+1).zfill(6) + ".png", im)
        res.append(im)
    return np.array(res)

def build_matrix(data_dir, num_im, channel, suff):
    mtx = []
    for i in range(1, num_im + 1):
        im = io.imread(data_dir + str(i).zfill(6) + suff)
        if (len(im.shape) == 2):
            im = np.where(im > 255/2, 1, 0)
            im = im.flatten()
        elif (channel == GRAYSCALE):
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).flatten()
        elif (channel == RED):
            im = im[:, :, 0].flatten()
        elif (channel == GREEN):
            im = im[:, :, 1].flatten()
        elif (channel == BLUE):
            im = im[:, :, 2].flatten()
        else:
            return None
        mtx.append(im)
    return np.array(mtx).T

def calc_romp(projector, camera):
    pass

'''
projector - (pq, k)
camera - (mn, k)
'''
def calculate_ltm(projector, camera, method=OMP):
    start = time.time()
    if (method == OMP):
        A = np.float32(projector.T)
        b = np.float32(camera.T)
        coef = orthogonal_mp(A, b, n_nonzero_coefs=150)
    elif (method == LEAST_SQUARES):
        coef, _, _, _ = np.linalg.lstsq(projector.T, camera.T, rcond=None)
        for i in range(coef.shape[0]):
            row = coef[i, :]
            thresh = (-row).argsort()[150]
            coef[i, :] = np.where(row >= thresh, row, 0)
    elif (method == ROMP):
        coef = calc_romp(projector, camera)
    else:
        mn, _ = camera.shape
        pq, _ = projector.shape
        coef = np.zeros((pq, mn))
    print("Iteration took", (time.time() - start)/60, "minutes")
    return coef.T
    
'''
T - (j, pq)
virtual_light - (j, 1)
'''
def get_virtual(T, virtual_light):
    return T.T @ virtual_light

def relight(T, light):
    return T @ light