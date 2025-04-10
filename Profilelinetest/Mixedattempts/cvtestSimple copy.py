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


def extractCurves(path):
    # Step 1: Read the image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10,6))
    plt.imshow(image, cmap='gray')
    plt.show()
    #image = ndimage.gaussian_filter(ima, sigma=1)
    # Step 2: Threshold the image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Make curves white
    plt.figure(figsize=(10,6))
    plt.imshow(binary, cmap='gray')
    plt.show()
    # Step 3: Skeletonize the binary image
    #skeleton = morphology.skeletonize(binary // 255, method='lee')  # Normalize binary to 0 and 1
    skeleton = morphology.thin(binary // 255)
    skeleton = (skeleton * 255).astype("uint8")  # Convert back to 8-bit for OpenCV
    
    plt.figure(figsize=(10,6))
    plt.imshow(skeleton, cmap='gray')
    plt.show()
    
    def is_endpoint(skeleton, x, y):
        # Count the number of neighbors (8-connectivity)
        neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255) - 1  # Subtract 1 to exclude the pixel itself
        return neighbors == 1  # Exactly 1 neighbor indicates an endpoint
    # Step 4: Identify intersection points
    def is_intersection(skeleton, x, y):
        neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255)-1
        return neighbors > 2  # More than 2 neighbors means it's an intersection

    height, width = skeleton.shape

    intersection_points = []   
    for y in range(0, height):
        for x in range(0, width):
            if skeleton[y, x] == 255 and is_intersection(skeleton, x, y):
                intersection_points.append((x, y))
    print(intersection_points)

    # Step 5: Trace branches starting from endpoints or intersections
    def trace_branch(skeleton, start_x, start_y):
        branch = [(start_x, start_y)]
        visited = set(branch)
        stack = [(start_x, start_y)]  # Use a stack for DFS

        while stack:
            x, y = stack.pop()
            neighbors = []
        
            # Check neighbors within valid boundaries
            for ny in range(max(0, y-1), min(height, y+2)):
                for nx in range(max(0, x-1), min(width, x+2)):
                    if (nx, ny) != (x, y) and skeleton[ny, nx] == 255 and (nx, ny) not in visited:
                        neighbors.append((nx, ny))
            for neighbor in neighbors:
                stack.append(neighbor)
                visited.add(neighbor)
                branch.append(neighbor)

        return branch

    branches = []
    for (x, y) in intersection_points:
        branch = trace_branch(skeleton, x, y)
        branches.append(branch)
    print(len(branches))
    # Step 6: Visualize skeleton and intersections
    skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for (x, y) in intersection_points:
        cv2.circle(skeleton_color, (x, y), 3, (0, 0, 255), -1)  # Mark intersections in red

    plt.figure(figsize=(10,6))
    plt.imshow(skeleton_color)
    plt.show()

    # Step 7: Return results
    # Step 6: Separate branches into groups
    group1 = branches[0]  # Assume the first branch corresponds to the first curve
    group2 = branches[1]  # Assume the second branch corresponds to the second curve

    # Step 7: Define a function to visualize each curve
    def plot_branch(branch, title):
        x_coords, y_coords = zip(*branch)  # Separate x and y coordinates
        plt.figure(figsize=(10, 5))
        plt.plot(x_coords, y_coords, 'b-')  # Plot the curve as a blue line
        plt.title(title)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.gca().invert_yaxis()  # Invert the y-axis to match image coordinates
        plt.show()

    # Step 8: Plot each curve in separate figures
    plot_branch(group1, "Curve 1")
    plot_branch(group2, "Curve 2")
    return branches

xtemp = 0
ytemp = 12000
wtemp = 1250

#testFile = "Profilelinetes/overlaytest0.tif"
testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Profilelinetest/Skeleton2.tif"
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
branches = extractCurves(testFile)

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