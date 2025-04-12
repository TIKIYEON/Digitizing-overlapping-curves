import imageio.v3 as iio
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_triangle
from skimage.morphology import disk, extrema
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb
from doctest import testfile
import math
from pathlib import Path
import imageio.v3 as iio
import cv2
#from numpy.ma.timer_comparison import cur
import skimage
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, opening, skeletonize
from skimage.transform import probabilistic_hough_line
import cv2

folderpathhisto = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/ProcessRootpainter/firstrootpainter.tif"

#temp = iio.plugins(folderpathhisto)
x = 0
y = 2500
w = 1250

image = iio.imread(uri=folderpathhisto)#[y:y+w, x:x+w-200,:]
fig, ax = plt.subplots()
ax.imshow(image)
print("plot 1:")
plt.show()
red_channel = image[..., 0]
redthreshmask = red_channel < 80

green_channel = image[..., 1]
greenthreshmask = green_channel > 150 

blue_channel = image[..., 2]
bluethreshmask = blue_channel > 150

combined_mask = redthreshmask & greenthreshmask & bluethreshmask

image2 = image.copy()
image2[~combined_mask] = [255,255,255,255]
fig, ax = plt.subplots()
ax.imshow(image2)
print("plot 1:")
plt.show()

image3 = image.copy()
image3[combined_mask] = [0,0,0,255]
fig, ax = plt.subplots()
ax.imshow(image3)
print("plot 1:")
plt.show()

I_grayg = skimage.color.rgb2gray(image2[...,3])
plt.figure(figsize=(10,6))
plt.imshow(I_grayg, cmap='gray')
plt.show()
#iio.imwrite("../testfolder/stampremoved2.tif", image3)



""" red_channel2 = red_channel.copy()
red_channel2[~combined_mask] = 235

green_channel2 = green_channel.copy()
green_channel2[~combined_mask] = 227

blue_channel2 = blue_channel.copy()
blue_channel2[~combined_mask] = 214

def concat_channels(r, g, b):
    assert r.ndim == 2 and g.ndim == 2 and b.ndim == 2
    return np.stack((r, g, b), axis=-1)

newimage = concat_channels(red_channel2, green_channel2, blue_channel2)
fig, ax = plt.subplots()
ax.imshow(newimage)
print("plot 1:")
plt.show()
 """
""" image = iio.imread(uri=folderpathhisto)[y:y+w, x:x+w-200,:]
red_channel = image[..., 0]
redthresh = red_channel > 190
green_channel = image[..., 1]
greenthresh = green_channel < 211
blue_channel = image[..., 2]
bluethresh = blue_channel < 211

def concat_channels(r, g, b):
    assert r.ndim == 2 and g.ndim == 2 and b.ndim == 2
    rgb = (r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis])
    return np.concatenate(rgb, axis=-1)

newimage = concat_channels(redthresh, greenthresh, bluethresh)

fig, ax = plt.subplots()
ax.imshow(newimage, cmap=['Reds','Blues','Greens'])
print("plot 1:")
plt.show()

 """