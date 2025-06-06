import math
from os import error
from pathlib import Path
import imageio.v3 as iio
import cv2
#from numpy.ma.timer_comparison import cur
import skimage
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, opening, skeletonize, remove_small_objects, thin
from skimage.transform import probabilistic_hough_line
from skimage.measure import label, regionprops
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

#src_path = 'Fundus_photograph_of_normal_left_eye.jpg'

def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()




testFile = "NonworkingMultipleintersections/T00105.las-1v0.tif"
#testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Multipleintersections/rotated_image.tif"
img = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE) # 0 imports a grayscale
if img is None:
    raise(ValueError(f"Image didn\'t load. Check that '{testFile}' exists."))
plt.imshow(img)
plt.show()
S = skeletonize(img > 128)
plt.imshow(S)
plt.show()
a, b = detect_ridges(S, sigma=3.0)

plot_images(S, a, b)

b = np.rot90(b)

plt.imshow(a, cmap='gray')
plt.show()

plt.imshow(b, cmap='gray')
plt.show()

def extractCurves(ima):
    #Read the image
    #ima = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #image = ndimage.gaussian_filter(img, sigma=1.0)
    
    plt.figure(figsize=(10,6))
    plt.imshow(ima, cmap='gray')
    plt.show()
    image = ndimage.gaussian_filter(ima, sigma=1.0)
    plt.figure(figsize=(10,6))
    plt.imshow(image, cmap='gray')
    plt.show()
    threshold = skimage.filters.threshold_otsu(image)
    #Threshold the image
    #binimage = np.where(125,255,0).astype(np.uint8)
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)  # Make curves white
    plt.figure(figsize=(10,6))
    plt.imshow(binary, cmap='gray')
    plt.show()
    #Skeletonize the binary image
    #dilated = cv2.dilate(binary, kernel=np.ones((3,3), np.uint8), iterations=1)
    """ eroded = cv2.erode(binary, kernel=np.ones((3,3), np.uint8), iterations=1)
    plt.figure(figsize=(10,6))
    plt.imshow(eroded, cmap='gray')
    plt.show() """
    #dilated = cv2.dilate(binary, kernel=np.ones((3,3), np.uint8), iterations=1)
    """ plt.figure(figsize=(10,6))
    plt.imshow(dilated, cmap='gray')
    plt.show() """
    skeleton = morphology.thin(binary)  # Normalize binary to 0 and 1
    skeleton = (skeleton * 255).astype("uint8")  # Convert back to 8-bit for OpenCV
    #skeleton = morphology.thin(binary)

    # Pruning the skeleton by removing endpoints iteratively

    #skeleton = morphology.thin(binary)
    
    
    #skeleton = ndimage.gaussian_filter(skeleto, sigma=1.0)
    plt.figure(figsize=(10,6))
    plt.imshow(skeleton, cmap='gray')
    plt.show()
    
    #Helper functions, finds all points with only 1 neighbour in 8 connectivity
    def is_endpoint(skeleton, x, y):
        # Count the number of neighbors (8-connectivity)
        neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255) - 1 # Subtract 1 to exclude the pixel itself
        return neighbors == 1  # Exactly 1 neighbor indicates an endpoint
    #Helper functions, finds all points with more than 2 neighbours in 8 connectivity
    def is_intersection(skeleton, x, y):
        neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255)-1
        return neighbors > 2  # More than 2 neighbors means it's an intersection

    #Finds all the endpoints and intersection points in skeletonized picture
    height, width = skeleton.shape
    print(height)
    print(width)
    
    intersection_points = []  
    iterations = 25
    for i in range(iterations):
        # Thinning step:
        thinned = morphology.skeletonize(skeleton > 0)  # ensure input is boolean
        thinned = (thinned.astype(np.uint8)) * 255

        # Endpoint detection on the thinned skeleton:
        endpoint_coords = []  # list of (x, y) tuples
        # Extract only foreground pixels for faster iteration
        foreground_pixels = np.argwhere(thinned == 255)  # Returns (y, x) pairs

        # Efficiently check for endpoints and intersections
        for y, x in foreground_pixels:
            if 1 <= y < height - 1 and 1 <= x < width - 1:  # Ensure we stay within bounds
                if is_endpoint(thinned, x, y):
                    endpoint_coords.append((x, y))

        # Create a mask for endpoints :
        endpoint_mask = np.zeros_like(thinned, dtype=np.uint8)
        for (x, y) in endpoint_coords:
            endpoint_mask[y, x] = 255

        # remove endpoints without dilation:
        for (x, y) in endpoint_coords:
            thinned[y, x] = 0
       
        # Remove endpoints using a dilated mask:
        """ kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(endpoint_mask, kernel, iterations=1)
        thinned = cv2.bitwise_and(thinned, cv2.bitwise_not(dilated_mask)) """

        # Update the skeleton for the next iteration:
        skeleton = thinned.copy()
        
        # Debug visualization (optional):
        """ plt.figure(figsize=(6, 4))
        plt.title(f"Iteration {i+1}")
        plt.imshow(skeleton, cmap='gray')
        plt.show() """
    plt.figure(figsize=(10,6))
    plt.title("pruned")
    plt.imshow(skeleton, cmap='gray')
    plt.show()
    """ for i in range(iterations): 
        original = skeleton.copy()
        x1 = morphology.skeletonize(original)
        plt.figure(figsize=(10,6))
        plt.title("x1")
        plt.imshow(x1, cmap='gray')
        plt.show()
        x1 = (x1 * 255).astype("uint8")
        x2 = []
        for y in range(0, height):
            for x in range(0, width):
                if x1[y, x] == 255 and is_endpoint(x1, x, y):
                    x2.append((x, y))
        
        #for i in range(len(endtemp)):
        #tempp = endtemp[i]
        mask = np.zeros_like(original, dtype=np.uint8)
        kernel1 = np.ones((3,3), np.uint8)
        for (x,y) in x2:
            mask[y,x] = 255
        
        temp_dilation = cv2.dilate(mask, kernel1,iterations=1)  # Apply dilation only on endpoints
        plt.figure(figsize=(10,6))
        plt.title("tempdialation")
        plt.imshow(temp_dilation, cmap='gray')
        plt.show()
        x3 = np.where(temp_dilation == 255, 255, original)
        plt.figure(figsize=(10,6))
        plt.title("x3")
        plt.imshow(x3, cmap='gray')
        plt.show()
        #dilated = cv2.dilate(skeleton, kernel= kernel1, iterations=1)
        
        x4 = np.logical_or(x1, x3)
        skeleton = (x4 * 255).astype("uint8")
        #skeleton = (skeleton * 255).astype("uint8")
        plt.figure(figsize=(10,6))
        plt.title("x4/skeleton")
        plt.imshow(skeleton, cmap='gray')
        plt.show() """
    """ kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(endtemp, kernel) """
    """ skeleton = np.where(original == 1, x3, original)
    plt.figure(figsize=(10,6))
    plt.imshow(skeleton, cmap='gray')
    plt.show() """  
    endpointstemp = []

    for y in range(0, height):
            for x in range(0, width):
                if skeleton[y,x] == 255 and is_endpoint(skeleton, x, y):
                    endpointstemp.append((x,y))
                if skeleton[y, x] == 255 and is_intersection(skeleton, x, y):
                    intersection_points.append((x, y))
    endpoints = sorted(endpointstemp, key=lambda point: point[0])

    
    
    #endpoints = sorted_points
    #Split the sorted array into two arrays
    #mid_index = len(sorted_points) // 2
    #startpoints = sorted_points[:mid_index]  # First half
    #endpoints = sorted_points[mid_index:]  # Second half   
    print(intersection_points)
    
    
    #print(startpoints)
    print(endpoints)
    #print(sorted_points)
    
    #If there´s intersections points, can only handle 1 right now
    if len(intersection_points) > 0:
        skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        for (x, y) in intersection_points:
            cv2.circle(skeleton_color, (x, y), 3, (0, 0, 255), -1)  # Mark intersections in red

        plt.figure(figsize=(10,6))
        plt.imshow(skeleton_color)
        plt.show()
        min_x = (min(point[0] for point in intersection_points) - 1)
        max_x = (max(point[0] for point in intersection_points) + 1)
    else:
        min_x = 0
        max_x = 0
    # Iterate and find all curve elements with specific x value
    def profileline(x):
        
        binary_points = []
        
        # Ensure x is within image bounds
        if 0 <= x < skeleton.shape[1]:
            # Loop over all y-coordinates in the column
            for y in range(skeleton.shape[0]):
                if skeleton[y, x] == 255:  # Binary point (white in binary image)
                    binary_points.append((x, y))  # Append the (x, y) position
       
        result = []
        # Handles if there´s 2 or more neighboring pixels on the profile line, it picks the one with the highest y value
        for i in range(len(binary_points)):
            # Checks if there's a previous point with same x value, and if it is +1
            if i > 0 and binary_points[i][1] == binary_points[i - 1][1] + 1:
                # Pop the previous point
                result.pop()  
            # Add the current point to the result
            result.append(binary_points[i])
        print(f"Binary points: {binary_points}")
        print(f"Final Binary points: {result}")
        return result
    intersectstart = profileline(min_x)
    intersectend = profileline(max_x)

    #Helper function that calculates a slope
    def checkslope(point1, point2):
        x = point2[0]-point1[0]
        if x == 0:
            return 0.0
        return (point2[1]-point1[1])/x
    
    #Helper function that interpolates the x value in between 2 points
    def interpolate(x, point1, point2):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]

        if x < x1 or x > x2:
            raise ValueError("x is out of bounds!")
        # Calculate the interpolated y-value
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    #Helper functions that rounds .5 and above to 1, and below to 0, to choose the closest matching pixel
    def col_round(x):
        frac = x - math.floor(x)
        if frac < 0.5: return math.floor(x)
        return math.ceil(x)
    # Refined trace function to handle intersections
    def trace_curve_with_gradients(skeleton, start_x, start_y, intersection_start, intesection_end):
        #sets up needed variables
        curve = [(start_x, start_y)]
        visited = set(curve)
        current_x, current_y = start_x, start_y
        localmaximaormin = (start_x,start_y)
        currenttempslope = 999
        counter = 0
        h, w = skeleton.shape
        #Threhsolds used for when wrapping, so it doesn´t interpolate if x is more than 1 pixel away
        height95thresh = (h/100.0)*95.0
        height5thresh = (h/100)*5
        while True:
            neighbors = []
            counter = counter+1
            #(10,10) range(9, 12) (9,9) (9,10) (9,11)
            #Finds all neighbors, except those with -1 x pixel values
            for ny in range(max(0, current_y-1), min(skeleton.shape[0], current_y + 2)):
                for nx in range(max(0, current_x), min(skeleton.shape[1], current_x + 2)):
                    
                    if (nx, ny) != (current_x, current_y) and skeleton[ny, nx] == 255 and (nx, ny) not in visited:
                        neighbors.append((nx, ny))
                #grad = (ny - current_y) / (nx - current_x + 1e-6)
                        
            #Termination needs to be here, but expanded to handle wrapping and non connected curves
            if not neighbors:
                #Termination of while true loop, if last endpoint has no neighbors, or no endpoints is left
                if len(endpoints) == 1 or len(endpoints) == 0:
                    endpoints.pop()
                    break
                
                #Get index number if found endpoint
                tempsave = 0
                temppoint = (current_x, current_y)
                for i in range(len(endpoints)):
                    if temppoint == endpoints[i]:
                        tempsave = i
                #Endpoints are sorted by x, so check all the endspoints with higher x value
                for j in range(tempsave,len(endpoints)):

                    tempendpoint = endpoints[j]
                    #If higher x value
                    if tempendpoint[0] > current_x:
                        #If no wrapping 
                        if not (current_y > height95thresh) and not (tempendpoint[1] < height5thresh):
                            if not (temppoint[1] > height95thresh) and not (current_y < height5thresh):  
                                #Linearly interpolate the space between the 2 found endpoints
                                numofpoints = tempendpoint[0] - current_x
                                for i in range(1,numofpoints):
                                    tempx = current_x+i
                                    ytemp = col_round(interpolate(tempx, temppoint, tempendpoint))
                                    #Avoid ending curve prematurely, by making sure original pixels non interpolated
                                    #Is added to visited
                                    for ny in range(max(0, ytemp-1), min(skeleton.shape[0], ytemp + 2)):
                                        for nx in range(max(0, tempx), min(skeleton.shape[1], tempx + 2)):
                                            if (nx, ny) != (current_x, current_y) and skeleton[ny, nx] == 255 and (nx, ny) not in visited:
                                                visited.add((nx, ny))
                                    visited.add((tempx, ytemp))
                                    curve.append((tempx, ytemp))   
                        #Set current x and y to new endpoint
                        current_x = tempendpoint[0]
                        current_y = tempendpoint[1]
                        #Remove the new endpoint from endpoints list 
                        endpoints.remove(endpoints[j])  
                        #break 
                        break
                #Remove the original endpoint from the endpoint list
                endpoints.remove(endpoints[tempsave])
                #Check for neighbors
                for ny in range(max(0, current_y-1), min(skeleton.shape[0], current_y + 2)):
                    for nx in range(max(0, current_x), min(skeleton.shape[1], current_x + 2)):
                        
                        if (nx, ny) != (current_x, current_y) and skeleton[ny, nx] == 255 and (nx, ny) not in visited:
                            neighbors.append((nx, ny))
                #If no neighbors, terminate
                if len(neighbors) == 0:
                    break
            #Counter used to make sure slope is updated enough
            counter = counter + 1
            #Checks if slope goes from positive to negative or reverse
            #If does, update the new localmaxormin, and reset the counter
            tempslope = checkslope((current_x,current_y), neighbors[0])
            if (tempslope != 0.0):
                if currenttempslope == 999:
                    currenttempslope = tempslope
                if (currenttempslope > 0 and tempslope > 0):
                    currenttempslope = tempslope
                if (currenttempslope < 0 and tempslope < 0):
                    currenttempslope = tempslope
                if (currenttempslope > 0 and tempslope < 0):
                    localmaximaormin = (current_x,current_y)
                    currenttempslope = tempslope
                    counter = 0
                    print(localmaximaormin)
                if (currenttempslope < 0 and tempslope > 0):
                    localmaximaormin = (current_x,current_y)
                    currenttempslope = tempslope
                    counter = 0
                    print(localmaximaormin)
            
            #Go to next neibor
            current_x, current_y = neighbors[0]  
            #If the next neighbor is in intersection
            if (current_x,current_y) in intersection_start:
                visited.add((current_x, current_y))
                curve.append((current_x, current_y))
                bestslope = 999999.9
                tempindex = 0
                point = (current_x,current_y)
                #tempstart = curve[0]
                curvejumpback = 10
                if counter < curvejumpback:
                    grad = checkslope(localmaximaormin,point)
                else:
                    grad = checkslope(curve[-curvejumpback],point)

                for i in range(0,len(intesection_end)):     
                    slope = checkslope(point, intesection_end[i])
                    compare = abs(grad-slope)
                    if compare < bestslope:
                        bestslope = compare
                        tempindex = i
                endpoint = intesection_end[tempindex] 
                numofpoints = endpoint[0] - point[0]
                for i in range(1,numofpoints):
                    tempx = current_x+i
                    ytemp = col_round(interpolate(tempx, point, endpoint))
                    for ny in range(max(0, ytemp-1), min(skeleton.shape[0], ytemp + 2)):
                        for nx in range(max(0, tempx), min(skeleton.shape[1], tempx + 2)):
                            
                            if (nx, ny) != (current_x, current_y) and skeleton[ny, nx] == 255 and (nx, ny) not in visited:
                                visited.add((nx, ny))
                    visited.add((tempx, ytemp))
                    curve.append((tempx, ytemp))

                current_x = endpoint[0]
                current_y = endpoint[1]  
                visited.add((current_x, current_y))
                curve.append((current_x, current_y))
            else:
                visited.add((current_x, current_y))
                curve.append((current_x, current_y))

        return curve

    curves = []
    #Start on first endpoint, and iterate through endpoints
    for x, y in endpoints: #+ intersection_points:
        curve = trace_curve_with_gradients(skeleton, x, y, intersectstart, intersectend)
        curves.append(curve)

    # Plot curve helper function
    def plot_curve(curve, title):
        x_coords, y_coords = zip(*curve)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_xlim(0.0, width)
        ax.set_ylim(0.0, height)
        #plt.figure(figsize=(10, 5))
        ax.plot(x_coords, y_coords, 'b-')
        plt.title(title)
        plt.gca().invert_yaxis()  # Match image coordinates
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    for i, curve in enumerate(curves):
        plot_curve(curve, f"Curve {i + 1}")
        #print(curve)
    return curves
    


xtemp = 0
ytemp = 12000
wtemp = 1250

#image = cv2.imread(testFile)
#image = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)
""" cv2.imwrite("testfolder/scantest.png", img) """
image = b
h,w = image.shape
#Choose the region of interest including excat boundries the graph
rx, ry, rw, rh = 0,0, w, h
#rx,ry,rw,rh = cv2.selectROI('Select The Complete and Exact Boundaries of Graph',image)
#graph = image#[ry:ry+rh,rx:rx+rw]
#cv2.destroyWindow('Select The Complete and Exact Boundaries of Graph')

#tempgraph = process_chunks(image.copy())


#Extract the curve points on the image
#image_path = "path_to_your_image.png"
curves = extractCurves(image)

""" curverotated = np.rot90(curves[0])
print(curverotated)
def plot_curve(curve):
    h, w = curve.shape
    x_coords, y_coords = zip(*curve)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_xlim(0.0, h)
    ax.set_ylim(0.0, w)
    #plt.figure(figsize=(10, 5))
    ax.plot(x_coords, y_coords, 'b-')
    
    plt.gca().invert_yaxis()  # Match image coordinates
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    for i, curve in enumerate(curves):
        plot_curve(curve, f"Curve {i + 1}")
        print(curve)
    return curves

plot_curve(curverotated) """
#iio.imwrite("Testresults/plot.tif",curvenum)
#Map curve (x,y) pixel points to actual data points from graph
curve_normalized1 = []
""" for cx, cy in curves[0]:
    x_value = np.float64((cx / rw) * (x_max - x_min) + x_min)
    y_value = np.float64((1 - cy / rh) * (y_max - y_min) + y_min)
    curve_normalized1.append([x_value, y_value]) """
#Lasiotester
import matplotlib.pyplot as plt
import numpy as np

#lasfn = "../T14502Las/T14502_02-Feb-07_JewelryLog.las"
lasfn = "T14502Las/T14502_02-Feb-07_JewelryLog.las"
import lasio
las = lasio.read(str(lasfn),ignore_header_errors=True)
#las = lasio.read(str(lasfn),encoding="cp866")
#las = lasio.read(str(lasfn),encoding="windows-1251")

headers=las.keys()
units = {}
for j in range(0, len(headers)):
     uval = las.curves[j].unit
     units[headers[j]] = uval

dataarr = las.data
metaheaders=las.well.keys()
metadata=[]
metadata.append({})
metadata.append({})
metadata.append({})


for j in range(0, len(metaheaders)):
     uval = las.well[j].unit
     metadata[0][metaheaders[j]] = uval

     tval = las.well[j].value
     metadata[1][metaheaders[j]] = str(tval)

     dval = las.well[j].descr
     metadata[2][metaheaders[j]] = str(dval)

print(metadata)
print(units)
depth = las['DEPT']
#y = np.flip(depth,0)
tempGAMM = las['GAMM']
x = 0
""" for i, k in enumerate(units):
     print(i,k)
     tempdic = las[k]
     x = np.array(tempdic)
     fig, ax = plt.subplots(figsize=(10,20))
     ax.plot(x, depth, color='red', linewidth=1)
     ax.invert_yaxis
     plt.show() """



x = np.array(depth[100:216])
y = np.array(tempGAMM[100:216]) 
wrapcounter = 0
#Enter the min and max values from the source graph here
y_min,y_max = 0.0, h
x_min,x_max = 0.0, w 

#Normalizes the data to chosen x and y bounds
curve_normalized1 = [[np.float64((cx/rw)*(x_max-x_min)+x_min),np.float64((1-cy/rh)*(y_max-y_min)+y_min)] for cx,cy in curves[0]]
curve_normalized1 = np.array(curve_normalized1)

#Handles wrapping
ythreshmin = ((y_max - y_min)/100)*5
ythreshmax = ((y_max - y_min)/100)*95
maxy = 0
for i in range(len(curve_normalized1)-1):
    curpoint = curve_normalized1[i].copy()
    curpointoriginal = curpoint.copy()
    nextpoint = curve_normalized1[i+1].copy()
    
    if wrapcounter != 0:
        curpoint[1] = (wrapcounter * y_max) + curpoint[1]
        curve_normalized1[i] = curpoint
    if curpoint[1] > maxy:
        maxy = curpoint[1]
    #difference = abs(curpoint[1] - nextpoint[1])
    
    """ if differencepre > ythreshmax:
        wrapcounter += 1 """
    if (curpointoriginal[1] > ythreshmax ) and (nextpoint[1] < ythreshmin):
        wrapcounter += 1

    if (curpointoriginal[1] < ythreshmin ) and (nextpoint[1] > ythreshmax):
        wrapcounter -= 1


temppoint = curve_normalized1[-1].copy()
temppoint[1] = (wrapcounter * y_max) + temppoint[1]
curve_normalized1[-1] = temppoint  
""" if (curpoint[1] < ythreshmax) or (curpoint[1] > ythreshmin):
    continue """   
    

""" for cx, cy in curves[0]:
    normalized_point = [
        np.float64((cx / rw) * (x_max - x_min) + x_min),
        np.float64((1 - cy / rh) * (y_max - y_min) + y_min)
    ]
    curve_normalized1.append(normalized_point)

curve_normalized1 = np.array(curve_normalized1) """
print(curve_normalized1)
np.savetxt("Multipleintersections/2darray.txt", curve_normalized1, fmt='%2f', delimiter=',')

#Plot the simulatedcurve
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlim(0, 10)
ax.set_ylim(0.0, maxy+ythreshmin)
ax.plot(curve_normalized1[:,0],curve_normalized1[:,1],'o-',linewidth=3)
ax.grid(True)
plt.show()

""" ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
ridges = ridge_filter.getRidgeFilteredImage(image) """