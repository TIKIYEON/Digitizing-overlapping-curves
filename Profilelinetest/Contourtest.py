from doctest import testfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from skimage.morphology import skeletonize


testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Profilelinetest/Simcurve8.tif"
# Load the image
image = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)

def scalecali(img):
  scale = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
# Threshold the image to create a binary image
  scale = 255-scale
  _, scale_bin = cv2.threshold(scale, 40, 255, cv2.THRESH_BINARY)

  x, y, w, h = cv2.boundingRect(scale_bin)
  calibration = 0.5 / w

  plt.imshow(cv2.rectangle(scale, cv2.boundingRect(scale_bin), (255, 255, 0), 2))
  return calibration

calibration = scalecali(testFile)
def fiberLen(img, calibration, plot=True):
    fiber = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #invert
    fiber = 255 - fiber
    #Dialate and erode
    fiber = cv2.dilate(fiber, None, iterations=1)
    fiber = cv2.erode(fiber,None, iterations=1)

    _, fiber_bin = cv2.threshold(fiber,127,255, cv2.THRESH_BINARY)
    height, width = fiber_bin.shape

    for i in range(height):
        for j in range(width):
            fiber_bin[i][j] = 1 if fiber_bin[i][j] == 255 else 0
    
    fiber_skel = skeletonize(fiber_bin)
    plt.figure(figsize=(10,6))
    plt.imshow(fiber_skel, cmap='gray')
    plt.show()
    fiber_skel2 = fiber_skel.astype(np.uint8)

    contours, hierarchy = cv2.findContours(fiber_skel2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    fiber_contours = [c for c in contours if cv2.arcLength(c, False) > 200]
    
    plt.figure(figsize=(10,6))
    plt.imshow(fiber_contours, cmap='gray')
    plt.show()
    measurement = []
    label_coord_x = []
    label_coord_y = []

    #get contour perimeter, divide it by 2 and multiply by calibration factor
    for i, cnt in enumerate(fiber_contours):
        measurement.append(float(cv2.arcLength(fiber_contours[i], False) / 2) * calibration)
        #get coordinates if plot is True
        if plot is True:
            label_coord_x.append(fiber_contours[i][0][0][0]) #get first pixel of contours
            label_coord_y.append(fiber_contours[i][0][0][1]) #get second pixel of contours
        
    #plot fiber measurements if plot is True
    if plot is True:
        fiber_copy = fiber.copy()
        #loop through measurement values
        for i, value in enumerate(measurement):
            text = "{:.2f}".format(value)
            x = label_coord_x[i]
            y = label_coord_y[i]
            #put measurement labels in image
            """ cv2.putText(fiber_copy, text = text, org = (x, y), 
                       fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                       fontScale = 1,
                       color = (150, 150, 150),
                       thickness = 2) """
        """ plt.imshow(fiber_copy)
        plt.show() """

        
    return [img, measurement]
    
    #fiber_skel = 
# Find contours in the binary image

fiberLen(testFile, calibration)
""" contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Combine all contour points
all_points = []
for contour in contours:
    for point in contour:
        x, y = point[0]
        all_points.append((x, y))

# Convert to a NumPy array
all_points = np.array(all_points)

# Sort points by x-coordinate for easier slicing
all_points = all_points[np.argsort(all_points[:, 0])]

# Define slicing parameters
num_slices = 20
x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
slice_width = (x_max - x_min) / num_slices

# Separate points slice by slice
curve1_points = []
curve2_points = []

for i in range(num_slices):
    # Define the bounds for this slice
    x_start = x_min + i * slice_width
    x_end = x_start + slice_width

    # Extract points within this slice
    slice_points = all_points[(all_points[:, 0] >= x_start) & (all_points[:, 0] < x_end)]

    # If two clusters are detected, assign them to different curves
    if len(slice_points) > 0:
        mean_y = np.mean(slice_points[:, 1])
        for point in slice_points:
            x, y = point
            if y < mean_y:  # Below mean goes to Curve 1
                curve1_points.append(point)
            else:  # Above mean goes to Curve 2
                curve2_points.append(point)

# Convert lists to NumPy arrays
curve1_points = np.array(curve1_points)
curve2_points = np.array(curve2_points)

# Plot the curves to verify correctness
plt.figure(figsize=(10, 5))
plt.plot(curve1_points[:, 0], curve1_points[:, 1], label="Extracted Curve 1")
plt.plot(curve2_points[:, 0], curve2_points[:, 1], label="Extracted Curve 2")
plt.show() """