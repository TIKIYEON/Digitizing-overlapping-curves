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
testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/ProcessRootpainter/firstrootpainter.tif"

def extractCurves(path):
    #Read the image
    ima = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #image = ndimage.gaussian_filter(img, sigma=1.0)
    
    plt.figure(figsize=(10,6))
    plt.imshow(ima, cmap='gray')
    plt.show()
    image = ndimage.gaussian_filter(ima, sigma=1.0)
    #Threshold the image
    _, binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)  # Make curves white
    plt.figure(figsize=(10,6))
    plt.imshow(binary, cmap='gray')
    plt.show()
    # Step 3: Skeletonize the binary image
    dilated = cv2.dilate(binary, kernel=np.ones((3,3), np.uint8), iterations=1)
    #skeleton = morphology.skeletonize(dilated // 255, method= 'lee')  # Normalize binary to 0 and 1
    skeleton = morphology.skeletonize(dilated)
    skeleton = (skeleton * 255).astype("uint8")  # Convert back to 8-bit for OpenCV

    #skeleton = ndimage.gaussian_filter(skeleto, sigma=1.0)
    plt.figure(figsize=(10,6))
    plt.imshow(skeleton, cmap='gray')
    plt.show()


ima = iio.imread(testFile)
plt.figure(figsize=(10,6))
plt.imshow(ima)
plt.show()
#image = np.rot90(ima, k = 3)
#iio.imwrite("ProcessRootpainter/Rootpainterfirgurefirst"".tif", image)
curves = extractCurves(testFile)