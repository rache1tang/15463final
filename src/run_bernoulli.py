from implementation import *
from PIL import Image, ImageEnhance, ImageOps

cam_dir = "../data/bernoulli/camera/"
proj_dir = "../data/bernoulli/projector/"
im_prefix = "IMG_"
im_suffix = ".JPG"
proj_suff = ".png"
black_im_num = 411
num_patterns = 450
m, n, p, q = 256, 256, 256, 256

left = 1253
top = 915
right = 2245
bottom = 1907

#generate_bernoulli(proj_dir)
#plt.imshow(io.imread(cam_dir + im_prefix + str(start_num) + im_suffix))
#plt.show()

## Process images ##
'''contrast_factor = 1.75
shift = 55'''

'''start_num = 9894
end_num = 9974
for i in range(start_num, end_num + 1):
    fname = cam_dir + im_prefix + str(i).zfill(4) + im_suffix
    im = Image.open(fname)
    im = ImageOps.exif_transpose(im)
    im = im.crop((left, top, right, bottom))
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(contrast_factor)
    im = np.array(im)
    im = np.where(im > shift, im - shift, 0)
    im = cv2.resize(im, (m, n))
    nname = cam_dir + str(i - start_num + 1).zfill(6) + im_suffix
    io.imsave(nname, im)
    os.remove(fname)'''

'''offset = end_num - start_num + 1
start_num = 9977
end_num = 9999
for i in range(start_num, end_num + 1):
    fname = cam_dir + im_prefix + str(i).zfill(4) + im_suffix
    im = Image.open(fname)
    im = ImageOps.exif_transpose(im)
    im = im.crop((left, top, right, bottom))
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(contrast_factor)
    im = np.array(im)
    im = np.where(im > shift, im - shift, 0)
    im = cv2.resize(im, (m, n))
    nname = cam_dir + str(i - start_num + 1 + offset).zfill(6) + im_suffix
    io.imsave(nname, im)
    os.remove(fname)'''

'''offset += end_num - start_num + 1
start_num = 1
end_num = 335
for i in range(start_num, end_num + 1):
    fname = cam_dir + im_prefix + str(i).zfill(4) + im_suffix
    im = Image.open(fname)
    im = ImageOps.exif_transpose(im)
    im = im.crop((left, top, right, bottom))
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(contrast_factor)
    im = np.array(im)
    im = np.where(im > shift, im - shift, 0)
    im = cv2.resize(im, (m, n))
    nname = cam_dir + str(i - start_num + 1 + offset).zfill(6) + im_suffix
    io.imsave(nname, im)
    os.remove(fname)
'''
'''offset = 439
start_num = 337
end_num = 399
for i in range(start_num, end_num + 1):
    fname = cam_dir + im_prefix + str(i).zfill(4) + im_suffix
    im = Image.open(fname)
    im = ImageOps.exif_transpose(im)
    im = im.crop((left, top, right, bottom))
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(contrast_factor)
    im = np.array(im)
    im = np.where(im > shift, im - shift, 0)
    im = cv2.resize(im, (m, n))
    nname = cam_dir + str(i - start_num + 1 + offset).zfill(6) + im_suffix
    io.imsave(nname, im)
    os.remove(fname)'''

'''offset = 502
start_num = 401
end_num = 410
for i in range(start_num, end_num + 1):
    fname = cam_dir + im_prefix + str(i).zfill(4) + im_suffix
    im = Image.open(fname)
    im = ImageOps.exif_transpose(im)
    im = im.crop((left, top, right, bottom))
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(contrast_factor)
    im = np.array(im)
    im = np.where(im > shift, im - shift, 0)
    im = cv2.resize(im, (m, n))
    nname = cam_dir + str(i - start_num + 1 + offset).zfill(6) + im_suffix
    io.imsave(nname, im)
    os.remove(fname)'''

'''im_shift = 1

im1 = io.imread(cam_dir + "000450.JPG")
im2 = io.imread(cam_dir + "000451.JPG")

plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()

im2[:, :, 0] = np.vstack((np.zeros((im_shift, n)), im2[:, :, 0]))[:-im_shift, :]
im2[:, :, 1] = np.vstack((np.zeros((im_shift, n)), im2[:, :, 1]))[:-im_shift, :]
im2[:, :, 2] = np.vstack((np.zeros((im_shift, n)), im2[:, :, 2]))[:-im_shift, :]

im2[:, :, 0] = np.hstack((np.zeros((m, im_shift)), im2[:, :, 0]))[:, :-im_shift]
im2[:, :, 1] = np.hstack((np.zeros((m, im_shift)), im2[:, :, 1]))[:, :-im_shift]
im2[:, :, 2] = np.hstack((np.zeros((m, im_shift)), im2[:, :, 2]))[:, :-im_shift]

plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()'''

'''im_shift = 6 # empirically determined
for i in range(451, 513):
    fname = cam_dir + str(i).zfill(6) + im_suffix
    im = io.imread(fname)
    im[:, :, 0] = np.hstack((im[:, :, 0], np.zeros((m, im_shift))))[:, im_shift:]
    im[:, :, 1] = np.hstack((im[:, :, 1], np.zeros((m, im_shift))))[:, im_shift:]
    im[:, :, 2] = np.hstack((im[:, :, 2], np.zeros((m, im_shift))))[:, im_shift:]
    io.imsave(fname, im)'''

'''im_shift = 3 # empirically determined
for i in range(502, 513):
    fname = cam_dir + str(i).zfill(6) + im_suffix
    im = io.imread(fname)
    im[:, :, 0] = np.hstack((im[:, :, 0], np.zeros((m, im_shift))))[:, im_shift:]
    im[:, :, 1] = np.hstack((im[:, :, 1], np.zeros((m, im_shift))))[:, im_shift:]
    im[:, :, 2] = np.hstack((im[:, :, 2], np.zeros((m, im_shift))))[:, im_shift:]
    io.imsave(fname, im)'''

'''im_shift = 1 # empirically determined
for i in range(311, 513):
    fname = cam_dir + str(i).zfill(6) + im_suffix
    im = io.imread(fname)
    im[:, :, 0] = np.hstack((im[:, :, 0], np.zeros((m, im_shift))))[:, im_shift:]
    im[:, :, 1] = np.hstack((im[:, :, 1], np.zeros((m, im_shift))))[:, im_shift:]
    im[:, :, 2] = np.hstack((im[:, :, 2], np.zeros((m, im_shift))))[:, im_shift:]
    io.imsave(fname, im)'''

## Calculate T ##
'''save_dir = "../data/bernoulli/T_lstsq/"
if (not os.path.exists(save_dir)):
    os.mkdir(save_dir)

proj = build_matrix(proj_dir, num_patterns, GRAYSCALE, proj_suff)
proj = np.where(proj == 0, -1, 1)
camera = build_matrix(cam_dir, num_patterns, GRAYSCALE, im_suffix)
step = m*n//8

for i in range(8):
    T = calculate_ltm(proj, camera[i*step : (i+1)*step, :], method=LEAST_SQUARES)
    np.save(save_dir + "T" + str(i).zfill(2) + ".npy", T)
    del T
    print("Done", (i+1), "/", 8)'''


## Generate Virtual Image ##
pattern = np.zeros((m, n))
step_m = m//4
step_n = n//4
for i in range(0, m, step_m):
    for j in range(0, n, step_n):
        if ((i + j) % 2 == 0):
            pattern[i:i+step_m, j:j+step_n] = np.ones((step_m, step_n))
pattern = pattern / np.linalg.norm(pattern, axis=0)
pattern = pattern.flatten()
data_dir = "../data/bernoulli/T_lstsq/"
mn = m*n
step = mn//8


'''vi = np.zeros(p*q)
for i in range(mn//step):
    vl = pattern[i*step:(i+1)*step].reshape(-1, 1)
    T = np.load(data_dir + "T" + str(i).zfill(2) + ".npy")
    lit = get_virtual(T, vl).flatten()
    vi += lit
    print("Calculated", i + 1, "/", mn//step)

fname = "../data/bernoulli/checker_lstsq.jpg"
io.imsave(fname, vi.reshape(p, q))'''

'''generated = np.zeros(m*n)
pattern = np.where(pattern == 0, -1, 1)
pattern = pattern.reshape(-1, 1)
for i in range(mn//step):
    T = np.load(data_dir + "T" + str(i).zfill(2) + ".npy")
    generated[i*step:(i+1)*step] = relight(T, pattern).flatten()
    print("Generated", i+1, "/", mn//step)
fname = "../data/bernoulli/checker_relight_lstsq.jpg"
io.imsave(fname, generated.reshape(m, n))'''