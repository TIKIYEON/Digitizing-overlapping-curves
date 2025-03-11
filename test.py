import numpy as np
import imageio.v3 as iio
import skimage as ski
import matplotlib.pyplot as plt
x = 30
y = 6730
w = 10
im_rgb = iio.imread(uri="stampremoved.tiff")#[y:y+w, x:90,:]
O = ski.color.rgb2gray(im_rgb)
plt.imshow(O, cmap="gray")
plt.show()