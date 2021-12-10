from implementation import *

cam_dir = "../data/fixed_pattern/camera/"
im_prefix = "IMG_"
im_suffix = ".JPG"
start_num = 8612
end_num = 8675
black_im_num = 8676

left = 1380
top = 134
right = 2492
bottom = 1246

static_thresh = 75

## Process images ##
for i in range(start_num, end_num + 1):
    fname = cam_dir + im_prefix + str(i) + im_suffix
    im = io.imread(fname)
    im = im[top:bottom, left:right]
    thresh = io.imread(cam_dir + im_prefix + str(black_im_num) + im_suffix)[top:bottom, left:right]
    im = np.where(im > thresh, im, 0)
    im = np.where(im > static_thresh, im, 0)
    im = cv2.resize(im, (256, 256))
    
    nname = cam_dir + str(i - start_num + 1).zfill(6) + im_suffix
    io.imsave(nname, im)
    os.remove(fname)

os.rename(cam_dir + im_prefix + str(black_im_num) + im_suffix, \
          cam_dir + "zero_im" + im_suffix)

