import numpy as np
import imageio.v3 as iio
import skimage as ski
import matplotlib.pyplot as plt
x = 30
y = 6730
w = 10
im_rgb = iio.imread(uri="T14502Las/T14502_02-Feb-07_JewelryLog.tiff")[y:y+w, x:90,:]
plt.imshow(im_rgb)
plt.show()