# %%
# read and plot the data
import numpy as np
from PIL import Image

def conv(arr, k):
    """
    vanilla python convolve
    :param arr: input data
    :param k: kernel
    :return: convolve result
    """
    h, w = arr.shape
    new_arr = np.zeros((h, w), dtype=arr.dtype)
    kl = len(k)

    tmp_arr = np.pad(arr, (1, 1))

    for i in range(h):
        for j in range(w):
            tmp = np.sum(np.multiply(tmp_arr[i:i + kl, j:j + kl], k))
            new_arr[i][j] = np.clip(tmp, 0, 255)
    return new_arr


example_arr = np.array([[10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0], ])
example_kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1], ])

# print(convolve2D(example_arr, example_kernel))

# %%
# image read
img = Image.open("3.png")
img_rgb = np.array(img.convert("RGB"))
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

kernel_b1 = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1], ])
kernel_b2 = np.array([[0, -1, 0],
                      [-1, 8, -1],
                      [0, -1, 0], ])

img_b1 = conv(B, kernel_b1)
img_b2 = conv(B, kernel_b2)

origin = np.hstack((R, G, B))
temp = np.hstack((B, img_b1, img_b2))

temp = np.vstack((origin, temp))
R = Image.fromarray(np.uint8(temp))
R.show()
R.save("R_conv.jpg")


# print(img_b2.shape)
# print(img_b1.shape)


# img_r = Image.fromarray(np.uint8(img_r))
# img_r.show()
# img_r.save("img_r.jpg")
#
# result_b1 = Image.fromarray(np.uint8(img_b1))
# result_b1.show()
# result_b1.save("img_b1.jpg")
#
# result_b2 = Image.fromarray(np.uint8(img_b2))
# result_b2.show()
# result_b2.save("img_b2.jpg")

# %%
