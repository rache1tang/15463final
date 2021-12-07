import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2

################################
#### IMPLEMENTATION CHOICES ####
################################
# display 512 x 512 on projector
# 

def generate_fixed_pattern(data_dir):
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
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
    

    
def subdivide(pattern, captured, frame_order):
    pass

'''j = 1
for i in range(13, 77):
    fname = "../data/fixed_pattern_cap/DSC_" + str(i).zfill(4) + ".JPG"
    new_name = "../data/fixed_pattern_cap/" + str(j).zfill(6) + ".jpg"
    im = io.imread(fname)
    os.remove(fname)
    io.imsave(new_name, im)
    j += 1'''

generate_fixed_pattern("../data/fixed_pattern/")