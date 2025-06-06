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
image = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)
#image = ndimage.gaussian_filter(img, sigma=1.0)

#image = ndimage.gaussian_filter(ima, sigma=1)
# Step 2: Threshold the image
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Apply edge detection
edges = cv2.Canny(binary, 50, 150)
plt.figure(figsize=(10,6))
plt.imshow(edges, cmap='gray')
plt.show()

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area
threshold_min = 1  # Adjust the minimum threshold value
threshold_max = 1000000000  # Adjust the maximum threshold value
filtered_contours = [cnt for cnt in contours if threshold_min < cv2.contourArea(cnt) < threshold_max]

# Create an empty image to draw the filtered contours
output = np.zeros_like(image)

# Draw the filtered contours
for cnt in filtered_contours:
    cv2.drawContours(output, [cnt], -1, 255, 1)

# Display the traced curves
plt.imshow(output, cmap='gray')
plt.title('Traced Curves')
plt.axis('off')
plt.show()