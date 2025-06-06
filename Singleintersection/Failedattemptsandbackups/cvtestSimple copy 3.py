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
testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Profilelinetes/Skeleton2.tif"

image = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5, 5), 0)  # Smooth the image
plt.figure(figsize=(10,6))
plt.imshow(image, cmap='gray')
plt.show()
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10,6))
plt.imshow(binary, cmap='gray')
plt.show()

kernel = np.ones((3, 3), np.float32)/3
dialate = cv2.dilate(binary, kernel, iterations=1)  # Merge nearby points
#binary = cv2.erode(binary, kernel, iterations=1)   # Refine the curves

#mask = binary2 > 200

#image2 = binary[mask] = [255]

skeleton = skeletonize(dialate // 255)  # Normalize binary image
skeleton = (skeleton * 255).astype("uint8")

plt.figure(figsize=(10, 6))
plt.imshow(skeleton, cmap='gray')
plt.title("Skeletonized Image with One Intersection")
plt.show()
xtemp = 0
ytemp = 12000
wtemp = 1250

#testFile = "Profilelinetes/overlaytest0.tif"

#image = cv2.imread(testFile)
#image = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)
""" cv2.imwrite("testfolder/scantest.png", img) """

#h,w,c = image.shape
#Choose the region of interest including excat boundries the graph
#rx, ry, rw, rh = 0,0, w, h
#rx,ry,rw,rh = cv2.selectROI('Select The Complete and Exact Boundaries of Graph',image)
#graph = image#[ry:ry+rh,rx:rx+rw]
#cv2.destroyWindow('Select The Complete and Exact Boundaries of Graph')

#tempgraph = process_chunks(image.copy())

#Enter the min and max values from the source graph here
y_min,y_max = 0.0, 1.0
x_min,x_max = 0.0, 10.0    

#Extract the curve points on the image
#image_path = "path_to_your_image.png"
#branches = extractCurves(testFile)

#iio.imwrite("Testresults/plot.tif",curvenum)
#Map curve (x,y) pixel points to actual data points from graph
""" curve_normalized = [[float((cx/rw)*(x_max-x_min)+x_min),float((1-cy/rh)*(y_max-y_min)+y_min)] for cx,cy in curve]
curve_normalized = np.array(curve_normalized)
print(curve_normalized)

#Plot the simulatedcurve
plt.figure(figsize=(15,7))
plt.plot(curve_normalized[:,0],curve_normalized[:,1],'o-',linewidth=3)
plt.title('Curve Re-Constructed')
plt.grid(True)
plt.show()
# Define the function
def curve_function1(x):
    return -0.001*x**3 + 0.0042*x**2 + 0.11*x

def curve_function2(x):
    return 0.001*x**3 - 0.0042*x**2 - 0.11*x + 1

# Generate 500 points for x between 0 and 10
x_values1 = np.linspace(0, 10, 500)
y_values1 = curve_function1(x_values1)

x_values2 = np.linspace(0, 10, 500)
y_values2 = curve_function2(x_values2)

#Same format for print
temparraycurve1 = np.zeros((len(x_values1),2))
for i in range(len(x_values1)):
    temparraycurve1[i] = (x_values1[i], y_values1[i])

testdatacurve1 = np.array(temparraycurve1)
print(testdatacurve1)

#Same format for print
temparraycurve2 = np.zeros((len(x_values2),2))
for i in range(len(x_values2)):
    temparraycurve2[i] = (x_values2[i], y_values2[i])

testdatacurve2 = np.array(temparraycurve2)
print(testdatacurve2)

from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Filter for NAN
testfiltered = testdatacurve1[~np.isnan(testdatacurve1).any(axis=1)]
# Extract x and y from test and real
x_sim, y_sim = curve_normalized[:, 0], curve_normalized[:, 1]
x_real, y_real = testfiltered[:, 0], testfiltered[:, 1]

# Interpolate real values
interpolator = interp1d(x_real, y_real, kind='linear', fill_value="extrapolate")
#Interpolate sim
y_real_interp = interpolator(x_sim)

#Mean Square and mean absolute
mse = mean_squared_error(y_sim, y_real_interp)
mae = mean_absolute_error(y_sim, y_real_interp)

print(mse)
print(mae) """