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

#folderpathhisto = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/ProcessRootpainter/firstrootpainter.tif"
#folderpathhisto2 = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/ProcessRootpainter/blueextracted.tif"
folderpathhisto2 = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Multipleintersections/rotated_image.tif"
#temp = iio.plugins(folderpathhisto)
x = 0
y = 2500
w = 1250

image = iio.imread(uri=folderpathhisto2)#[y:y+w, x:x+w-200,:]
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
#iio.imwrite("ProcessRootpainter/blueextracted.tif", image2)

image3 = image.copy()
image3[combined_mask] = [0,0,0,255]
fig, ax = plt.subplots()
ax.imshow(image3)
print("plot 1:")
plt.show()

I_grayg = cv2.imread(folderpathhisto2, cv2.IMREAD_GRAYSCALE)
#I_grayg = skimage.color.rgb2gray(folderpathhisto2)
plt.figure(figsize=(10,6))
plt.imshow(I_grayg, cmap='gray')
plt.show()
image = ndimage.gaussian_filter(I_grayg, sigma=1.0)
_, binary = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
""" th3 = cv2.adaptiveThreshold(I_grayg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,5) """
plt.figure(figsize=(10,6))
plt.imshow(binary, cmap='gray')
plt.show()
""" dilated = cv2.dilate(th3, kernel=np.ones((3,3), np.uint8), iterations=1)
plt.figure(figsize=(10,6))
plt.imshow(dilated, cmap='gray')
plt.show() """
#skeleton = morphology.skeletonize(dilated // 255, method= 'lee')  # Normalize binary to 0 and 1
skeleton = morphology.skeletonize(binary)
skeleton = (skeleton * 255).astype("uint8")  # Convert back to 8-bit for OpenCV

#skeleton = ndimage.gaussian_filter(skeleto, sigma=1.0)
plt.figure(figsize=(10,6))
plt.imshow(skeleton, cmap='gray')
plt.show()

""" image = ndimage.gaussian_filter(I_grayg, sigma=1)
plt.figure(figsize=(10,6))
plt.imshow(image, cmap='gray')
plt.show()
#Threshold the image
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Make curves white
plt.figure(figsize=(10,6))
plt.imshow(binary, cmap='gray')
plt.show() """


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