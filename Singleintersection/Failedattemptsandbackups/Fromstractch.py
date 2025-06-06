## Libraries

from pathlib import Path
import imageio.v3 as iio
import cv2
import skimage
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, opening, skeletonize
testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Profilelinetes/Simcurve8.tif"
# Load the image
image = iio.imread(testFile)
image2 = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)

I_gray = ndimage.gaussian_filter(image2, sigma=1)
plt.figure(figsize=(10,6))
plt.imshow(I_gray, cmap='gray')
plt.show()

_, I_grayg = cv2.threshold(I_gray, 127, 255, cv2.THRESH_BINARY_INV)
threshold = skimage.filters.threshold_otsu(I_grayg)
plt.figure(figsize=(10,6))
plt.imshow(I_grayg, cmap='gray')
plt.show()

overlays = []
mask = (I_grayg > threshold)
min_pixels = 200
row_mask = np.sum(mask, axis=1) < min_pixels
col_mask = np.sum(mask, axis=0) < min_pixels
#print(f'row mask: \n {row_mask}')
# Add a new axis and add the maskings to that as well with np.repeat
mask_combined = np.repeat((row_mask[:, np.newaxis] | col_mask[np.newaxis, :])[:, :, np.newaxis], image2.shape[2], axis=2)
#print(f'mask combined: \n {mask_combined}')
K = np.maximum(image2, np.max(image2) * mask_combined)
se = disk(10)
J = skimage.morphology.opening(K.mean(axis=2), se)
#J_closed = skimage.morphology.closing(J, se)
S = skeletonize(J<128)
#plt.figure(figsize=(10,10))
#plt.subplot(2, 2, 4)

overlay = np.zeros(image2.shape[:2] + (3,), dtype=np.uint8) + 255
overlay[S] = [255, 0, 0]
#plt.imshow(overlay)
#plt.imshow(i, cmap="gray", alpha=0.2)
#plt.axis("image")
print(overlay)
overlays.append(overlay)
print(overlays)

plt.imshow(overlay)
#plt.imshow(i, cmap="gray", alpha=0.2)
plt.axis("image")
plt.show()

