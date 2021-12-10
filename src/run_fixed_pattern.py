from implementation import *

cam_dir = "../data/fixed_pattern/camera/"
proj_dir = "../data/fixed_pattern/projector/"
im_prefix = "IMG_"
im_suffix = ".JPG"
proj_suff = ".jpg"
start_num = 8612
end_num = 8675
black_im_num = 8676
num_patterns = 64

left = 1380
top = 134
right = 2492
bottom = 1246

static_thresh = 75

## Process images ##
'''for i in range(start_num, end_num + 1):
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
'''

## Calculate T ##
'''save_dir = "../data/fixed_pattern/T/"
if (not os.path.exists(save_dir)):
    os.mkdir(save_dir)

proj = build_matrix(proj_dir, num_patterns, GRAYSCALE, proj_suff)
camera = build_matrix(cam_dir, num_patterns, GRAYSCALE, im_suffix)

mn, k = camera.shape
step = mn//8

for i in range(0, mn, step):
    T = calculate_ltm(proj, camera[i:i+step, :]) # the first j rows of T
    fname = save_dir + str(i).zfill(2) + ".npy"
    np.save(fname, T)
    print("Saved T", (i+1)//step, "/", mn//step)
'''