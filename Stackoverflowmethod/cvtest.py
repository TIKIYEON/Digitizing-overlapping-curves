## Libraries

from pathlib import Path
import imageio.v3 as iio
import cv2
import numpy
import skimage
import matplotlib.pyplot as pyplot

## Read and display the image
#Function to take difference between two consecutive points
now = 0
def differentiator(variable):
    global now
    before = now
    now = variable
    return now-before

#Function to find rising edges from profile data
def findRisingEdges(profile, min_height=1, max_height=255, min_width=1, max_width=255,smoothing=10):
    profile_diff = [differentiator(p) for p in profile]
    if smoothing:
        profile_diff = [pd if abs(pd)>smoothing else 0 for pd in profile_diff]
    rising_edges = []
    
    i=0
    while i < len(profile_diff): # do not check the last point
        edge_start,edge_end = None,None
        diff = profile_diff[i]

        #Search positive derivative for rising edge
        if diff > 0:
            edge_start = i-1 if i>1 else i

            #Find where the edge ends by searching non-positive derivative
            if i == (len(profile_diff)-1):# if its the last point then it is edge end
                edge_end = i
                i = i + 1
            else:
                while profile_diff[i+1]>0:
                    i = i + 1
                    if i == (len(profile_diff)-1):
                        break
                edge_end = i
                i = i + 1

            edge_width = edge_end - edge_start
            edge_height = profile[edge_end] - profile[edge_start]

            if edge_width >= min_width and edge_width <= max_width and edge_height >= min_height and edge_height <= max_height:
                rising_edges.append([edge_start,edge_end,edge_width,edge_height])
        else:
            i = i + 1

    return rising_edges

#Function to find falling edges from profile data
def findFallingEdges(profile, min_height=1, max_height=255, min_width=1, max_width=255,smoothing=10):
    profile_diff = [differentiator(p) for p in profile]
    if smoothing:
        profile_diff = [pd if abs(pd)>smoothing else 0 for pd in profile_diff]

    falling_edges = []
    i=0
    while i < len(profile_diff): # do not check the last point
        edge_start,edge_end = None,None
        diff = profile_diff[i]
        #Search negative derivative for rising edge
        if diff < 0:
            edge_start = i-1 if i>1 else i

            #Find where the edge ends by searching non-negative derivative
            if i == (len(profile_diff)-1):# if its the last point then it is edge end
                edge_end = i
                i = i + 1
            else:
                while profile_diff[i+1]<0:
                    i = i + 1
                    if i == (len(profile_diff)-1):
                        break
                edge_end = i
                i = i + 1
            
            edge_width = edge_end - edge_start
            edge_height = abs(profile[edge_end] - profile[edge_start])

            if edge_width >= min_width and edge_width <= max_width and edge_height >= min_height and edge_height <= max_height:
                falling_edges.append([edge_start,edge_end,edge_width,edge_height])
        else:
            i = i + 1

    return falling_edges

#Function to find sinks on the profile by looking for falling edges followed by rising edges
def findSinks(profile, min_width=3, min_depth=50, smoothing=10,
                min_fe_height=1, max_fe_height=255, min_fe_width=1, max_fe_width=255, 
                min_re_height=1, max_re_height=255, min_re_width=1, max_re_width=255):
    
    rEdges = findRisingEdges(profile, min_height=min_fe_height, max_height=max_fe_height, min_width=min_fe_width, max_width=max_fe_width,smoothing=smoothing)
    fEdges = findFallingEdges(profile, min_height=min_re_height, max_height=max_re_height, min_width=min_re_width, max_width=max_re_width,smoothing=smoothing)

    sinks = []
    for rising_edge in rEdges:
        rising_edge_start,rising_edge_end,reWidth,reHeight = rising_edge 
        falling_edge_starts = [f[0] for f in fEdges]

        #Find all the sinks
        for y in reversed(range(0,rising_edge_end)):

            if y in falling_edge_starts:
                falling_edge_start,falling_edge_end,feWidth,feHeight = fEdges[falling_edge_starts.index(y)]
                
                sinkwidth = rising_edge_end-falling_edge_start+1
                depth = (feHeight+reHeight)//2
                if sinkwidth>=min_width and depth>=min_depth:
                    sinks.append([falling_edge_start,rising_edge_end,depth])
                break
    return sinks

#Function to find the curve from the image 
def extractCurve(src_image, xmin, xmax):#profile_interval=5):
    #Read the image
    
    profile_interval = 5
    h,w,c = src_image.shape
    #profile_interval = int(w/(4*(xmax - xmin)))
    gray = cv2.cvtColor(src_image,cv2.COLOR_BGR2GRAY)

    #Scan the image through it's width(x-axis) get profile line extract curve's points
    last_y_level = 0 #this will always contain the latest curve point
    curve_points = []
    for xi in range(w//profile_interval):
        #Draw the profile on image
        cv2.line(graph,(xi*profile_interval,0),(xi*profile_interval,h),(255,0,0),1)

        #Get the profile
        profile = skimage.measure.profile_line(gray, (0,xi*profile_interval), (h,xi*profile_interval), linewidth=1, mode='constant') #Take the profile line

        #Find all the sinks in the profile data one of thesee points belongs to the curve
        sinks = findSinks(profile,smoothing=10,min_fe_height=150,min_re_height=150)

        if len(sinks)==0:   #If no sink no curve point
            pass
        
        elif len(sinks)==1: #If 1 sink it is the curve point
            start,end,depth = sinks[0]
            pX,pY = xi*profile_interval,(start+end)//2
            cv2.circle(src_image,(pX,pY),4,(0,0,255),-1)
            curve_points.append([pX,pY])
            last_y_level = pY
        
        else:   #If multiple sinks choose the one closest to last curve point
            closest = sinks[0]
            start,end,depth = closest
            min_y_dist = abs((start+end)//2 - last_y_level)
            sinks.pop(0)
            for sink in sinks:
                start,end,depth = sink
                y_dist = abs((start+end)//2 - last_y_level)
                if y_dist < min_y_dist:
                    min_y_dist = y_dist
                    closest = sink

            start,end,depth = closest
            pX,pY = xi*profile_interval,(start+end)//2
            cv2.circle(src_image,(pX,pY),4,(0,0,255),-1)
            curve_points.append([pX,pY])
            last_y_level = pY

        cv2.imshow('graph',graph)
        cv2.waitKey(1)
        # pyplot.plot(profile,'o-')
        # pyplot.show()

    return curve_points

def rotate_image(image, angle):
  image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

#Read the image
xtemp = 0
ytemp = 12000
wtemp = 1250
#imgFile = "T14502Las/T14502_02-Feb-07_JewelryLog-Kopi.tiff"
#testFile = "../Profilelinetes/overlaytest1.tif"
#testFile = "../testfolder/v2checktest2.tif"
#testFile = "../testresults/redscan.png"
#img = cv2.imread(imgFile)
""" img = cv2.imread(imgFile)[ytemp:ytemp+750, xtemp:xtemp+850,:]

image = rotate_image(img, 90) """
testFile = "NonworkingMultipleintersections/rotated_image.tif"
image = cv2.imread(testFile)
""" cv2.imwrite("testfolder/scantest.png", img) """
""" image = imag[xtemp:xtemp+wtemp-400, ytemp:ytemp+wtemp+400,:]
 """

h,w,c = image.shape
#Choose the region of interest including excat boundries the graph
#, ry, rw, rh = 0,0, w, h
rx,ry,rw,rh = cv2.selectROI('Select The Complete and Exact Boundaries of Graph',image)
graph = image[ry:ry+rh,rx:rx+rw]
cv2.destroyWindow('Select The Complete and Exact Boundaries of Graph')

#Enter the min and max values from the source graph here
y_min,y_max = 0, 200
x_min,x_max = 10604.7500, 10667.0   

#Extract the curve points on the image
curve = extractCurve(graph, x_min, x_max)

#iio.imwrite("Testresults/plot.tif",curvenum)
#Map curve (x,y) pixel points to actual data points from graph
curve_normalized = [[int((cx/rw)*(x_max-x_min)+x_min),int((1-cy/rh)*(y_max-y_min)+y_min)] for cx,cy in curve]
curve_normalized = numpy.array(curve_normalized)
print(curve_normalized)

#Plot the newly constructed curve
pyplot.figure(figsize=(15,7))
pyplot.plot(curve_normalized[:,0],curve_normalized[:,1],'o-',linewidth=3)
#pyplot.savefig('../testfolder/plot.tif', format='tiff', dpi=300)
pyplot.title('Curve Re-Constructed')
pyplot.grid(True)
pyplot.show()
