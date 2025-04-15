#from numpy.ma.timer_comparison import cur
import skimage
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, opening, skeletonize
from skimage.transform import probabilistic_hough_line
import cv2

testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/testfolder/scantestrotated.png"
# Load the image
image = cv2.imread(testFile)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Denoise the image using a morphological opening operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
plt.figure(figsize=(10,6))
plt.imshow(gray, cmap='gray')
plt.show()
morph_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
plt.figure(figsize=(10,6))
plt.imshow(morph_open, cmap='gray')
plt.show()
# Subtract the denoised image from the original to isolate high-frequency components (like curves)
subtracted = cv2.subtract(gray, morph_open)
plt.figure(figsize=(10,6))
plt.imshow(subtracted, cmap='gray')
plt.show()
# Apply Gaussian Blur to smooth the image
smoothed = cv2.GaussianBlur(subtracted, (5, 5), 0)
plt.figure(figsize=(10,6))
plt.imshow(smoothed, cmap='gray')
plt.show()
# Apply Adaptive Thresholding
binary = cv2.adaptiveThreshold(
    smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
plt.figure(figsize=(10,6))
plt.imshow(binary, cmap='gray')
plt.show()
# Perform edge detection
edges = cv2.Canny(binary, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area or shape (if known characteristics exist for the curves)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

# Create a blank mask
mask = np.zeros_like(gray)

# Draw the filtered contours onto the mask
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Combine the mask with the original image to isolate curves
isolated_curves = cv2.bitwise_and(image, image, mask=mask)

# Save or display the result
cv2.imwrite('improved_isolated_curves.png', isolated_curves)
cv2.imshow('Improved Isolated Curves', isolated_curves)
cv2.waitKey(0)
cv2.destroyAllWindows()