from implementation import *

cam_dir = "../data/bernoulli/camera/"
proj_dir = "../data/bernoulli/projector/"
im_prefix = "IMG_"
im_suffix = ".JPG"
proj_suff = ".png"
start_num = 8677
end_num = 8777
num_patterns = 64
m, n, p, q = 256, 256, 256, 256

left = 1380
top = 134
right = 2492
bottom = 1246

#generate_bernoulli(proj_dir)

## Process images ##
'''for i in range(start_num, end_num + 1):
    fname = cam_dir + im_prefix + str(i) + im_suffix
    im = io.imread(fname)
    im = im[top:bottom, left:right]
    im = cv2.resize(im, (m, n))
    nname = cam_dir + str(i - start_num + 1).zfill(6) + im_suffix
    io.imsave(nname, im)
    os.remove(fname)'''


## Calculate T ##
'''save_dir = "../data/fixed_pattern/T/"
if (not os.path.exists(save_dir)):
    os.mkdir(save_dir)

proj = build_matrix(proj_dir, num_patterns, GRAYSCALE, proj_suff)
camera = build_matrix(cam_dir, num_patterns, GRAYSCALE, im_suffix)
step = m*n//8

for i in range(8):
    T, b = calculate_ltm(proj, camera[i*step : (i+1)*step, :])
    np.save(save_dir + "T" + str(i).zfill(2) + ".npy", T)
    np.save(save_dir + "b" + str(i).zfill(2) + ".npy", b)
    del T
    del b
    print("Done", (i+1), "/", 8)'''


## Generate Virtual Image ##
'''pattern = np.zeros((m, n))
step_m = m//4
step_n = n//4
for i in range(0, m, step_m):
    for j in range(0, n, step_n):
        if ((i + j) % 2 == 0):
            pattern[i:i+step_m, j:j+step_n] = np.ones((step_m, step_n))
pattern = pattern / np.linalg.norm(pattern, axis=0)
pattern = pattern.flatten()
data_dir = "../data/fixed_pattern/T/"
mn = m*n
step = mn//8'''


'''vi = np.zeros(p*q)
for i in range(mn//step):
    vl = pattern[i*step:(i+1)*step].reshape(-1, 1)
    T = np.load(data_dir + "T" + str(i).zfill(2) + ".npy")
    b = np.load(data_dir + "b" + str(i).zfill(2) + ".npy")
    vi[:] += get_virtual(T, vl - b.reshape(-1, 1)).flatten()
    print("Calculated", i + 1, "/", mn//step)

fname = "../data/fixed_pattern/checker.jpg"
io.imsave(fname, vi.reshape(p, q))'''

'''generated = np.zeros(m*n)
pattern = pattern.reshape(-1, 1)
for i in range(mn//step):
    T = np.load(data_dir + "T" + str(i).zfill(2) + ".npy")
    b = np.load(data_dir + "b" + str(i).zfill(2) + ".npy")
    generated[i*step:(i+1)*step] = relight(T, pattern).flatten() + b
    print("Generated", i+1, "/", mn//step)
fname = "../data/fixed_pattern/checker_relight.jpg"
io.imsave(fname, generated.reshape(m, n))'''