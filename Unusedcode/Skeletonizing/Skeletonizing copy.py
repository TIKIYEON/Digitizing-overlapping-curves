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

#

#testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/testfolder/chunks/acombinedchunk.tif"
#testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/testfolder/scantest.png"
#testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/testfolder/stampremoved2.tif"
testFile2 = "T14502Las/T14502_02-Feb-07_JewelryLog.tiff"
testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/T12073/1930430DualPropResistivityGammaRayLogScan.tif"

def extractCurvesoriginal(image):
    array = []
    overlays = []
    plt.figure(figsize=(10,6))
    plt.imshow(image)
    plt.show()
    I_gray = skimage.color.rgb2gray(image)
    plt.figure(figsize=(10,6))
    plt.imshow(I_gray, cmap='gray')
    plt.show()
    #I_grayg = cv2.imread(temp, cv2.IMREAD_GRAYSCALE)
    # Grayscale
    #print(f'i.shape: \n {i.shape[2]}')
    #print(f'i gray.shape: \n {I_gray.shape}')
    I_grayg = ndimage.gaussian_filter(I_gray, sigma=1.0)
    plt.figure(figsize=(10,6))
    plt.imshow(I_grayg, cmap='gray')
    plt.show()
    h, w = I_grayg.shape
    # Attempt with OTSU's method
    threshold = skimage.filters.threshold_otsu(I_grayg)
    mask = (I_grayg > threshold)
    min_pixels = 200
    row_mask = np.sum(mask, axis=1) < w*0.8#min_pixels
    col_mask = np.sum(mask, axis=0) < h*0.8#min_pixels
    #print(f'row mask: \n {row_mask}')
    # Add a new axis and add the maskings to that as well with np.repeat
    mask_combined = np.repeat((row_mask[:, np.newaxis] | col_mask[np.newaxis, :])[:, :, np.newaxis], image.shape[2], axis=2)
    #print(f'mask combined: \n {mask_combined}')
    K = np.maximum(image, np.max(image) * mask_combined)
    plt.figure(figsize=(10,6))
    plt.imshow(K, cmap='gray')
    plt.show()
    se = disk(10)
    plt.figure(figsize=(10,6))
    plt.imshow(se, cmap='gray')
    plt.show()
    J = opening(K.mean(axis=2), se)
    S = skeletonize(J<128)
    #plt.figure(figsize=(10,10))
    #plt.subplot(2, 2, 4)
    overlay = np.zeros(image.shape[:2] + (3,), dtype=np.uint8) + 255
    overlay[S] = [255, 0, 0]
    #plt.imshow(overlay)
    #plt.imshow(i, cmap="gray", alpha=0.2)
    #plt.axis("image")
    overlays.append(overlay)
    if array == []:
        array = overlay
    else:
        array = np.concatenate((array,overlay),axis=0)
    
    plt.imshow(overlay, cmap='gray')
    #plt.imshow(i, cmap="gray", alpha=0.2)
    plt.show()

def extractCurves(path):
    array = []
    overlays = []
    image = iio.imread(path)
    I_grayg = skimage.color.rgb2gray(image)
    plt.figure(figsize=(10,6))
    plt.imshow(I_grayg, cmap='gray')
    plt.show()
    #I_grayg = cv2.imread(temp, cv2.IMREAD_GRAYSCALE)
    # Grayscale
    #print(f'i.shape: \n {i.shape[2]}')
    #print(f'i gray.shape: \n {I_gray.shape}')
    #I_grayg = ndimage.gaussian_filter(I_grayg, sigma=1)
    h, w = I_grayg.shape
    # Attempt with OTSU's method
    threshold = skimage.filters.threshold_otsu(I_grayg)-0.08
    mask = (I_grayg > threshold)
    min_pixels = 200
    row_mask = np.sum(mask, axis=1) < int(I_grayg.shape[1]*0.15)
    col_mask = np.sum(mask, axis=0) < int(I_grayg.shape[1]*0.15)
    #print(f'row mask: \n {row_mask}')
    # Add a new axis and add the maskings to that as well with np.repeat
    mask_combined = np.repeat((row_mask[:, np.newaxis] | col_mask[np.newaxis, :])[:, :, np.newaxis], image.shape[2], axis=2)
    #print(f'mask combined: \n {mask_combined}')
    K = np.maximum(image, np.max(image) * mask_combined)
    se = disk(10)
    J = opening(K.mean(axis=2), se)
    S = skeletonize(J<200)
    #plt.figure(figsize=(10,10))
    #plt.subplot(2, 2, 4)
    overlay = np.zeros(image.shape[:2] + (3,), dtype=np.uint8) + 255
    overlay[S] = [255, 0, 0]
    #plt.imshow(overlay)
    #plt.imshow(i, cmap="gray", alpha=0.2)
    #plt.axis("image")
    overlays.append(overlay)
    if array == []:
        array = overlay
    else:
        array = np.concatenate((array,overlay),axis=0)
    
    plt.imshow(overlay, cmap='gray')
    #plt.imshow(i, cmap="gray", alpha=0.2)
    plt.show()



#ima = iio.imread(testFile)
x = 90
y = 3000
""" image = np.rot90(ima, k = 3)
iio.imwrite("Skeletonizing/Acombinedchunkrot90.tif", image) """
image = iio.imread(testFile)[y:y+7500, x: x+760:]
curves = extractCurvesoriginal(image)

O = iio.imread(testFile2)
headerless_O = O[6700:14570]
headerless_O2 = headerless_O[:, 50:800]

extractCurves()