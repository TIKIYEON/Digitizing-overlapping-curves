from email.mime import image
from pathlib import Path
import imageio.v3 as iio
import cv2
import skimage
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, opening, skeletonize
import random

testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Profilelinetest/muVNT2.tif"

def distance_to_line(line_start, line_end, point):
  normalLength = np.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
  distance = np.float64(((point[0] - line_start[0])* (line_end[1]) - (point[1] - line_start[1])* (line_end[0]-line_start[0])))/ normalLength
  return np.abs(distance)

def splitcurve(img):
  image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
  #imag = ndimage.gaussian_filter(im, sigma=1.0)
  #_, image = cv2.threshold(imag, 127, 255, cv2.THRESH_BINARY_INV)
  plt.figure(figsize=(10,6))
  plt.imshow(image)
  plt.show()
  h, w = image.shape
  clusters = []
  maxDist = 10.0
  for x in range(0, w, 6):
    active = False
    ccluster = np.array([0,0])
    clustersupport = 0
    for y in range(0, h):
      cpoint = ((x,y))
      cactive = image[y,x] == 0
      if (cactive):
        ccluster += cpoint
        clustersupport +=1
        active = cactive
      
      if (active and not cactive):
        finishedcluster = (1.0 / clustersupport) * ccluster

        ccluster = np.array([0,0])
        clustersupport = 0
        active = False

        #adding cluster to list
        bestclust = -1
        bestDist = 99999999.99
        for i in range(0, len(clusters)):
          disttoclusters = 99999999.99
          if (len(clusters[i]) == 1):
            cdist = np.float64(cv2.norm(np.array(finishedcluster) - np.array(clusters[i][-1])))
            if (cdist < disttoclusters):
              disttoclusters = cdist
        
          else:
            lineA = clusters[i][len([i])-1]
            lineB = clusters[i][len([i])-2]
            #lineB = np.array(finishedcluster)
            cdist = distance_to_line(lineA,lineB, finishedcluster)
            if (cdist < disttoclusters):
              disttoclusters = cdist
          
          for j in range(0, len(clusters[i]), 1):
            cdist = np.float32(cv2.norm(np.array(finishedcluster) - np.array(clusters[i][-1])))
            if (cdist < disttoclusters):
              disttoclusters = cdist
          
          if (disttoclusters < maxDist):
            if (disttoclusters < bestDist):
              bestDist = disttoclusters
              bestclust = i
        
        if bestclust < 0:
            # Create a new cluster and addto it
            newCluster = [finishedcluster]  # In Python, lists are used instead of vectors
            clusters.append(newCluster)  # Append the new cluster to clusters
        else:
            # Add to the existing cluster
            clusters[bestclust].append(finishedcluster)
  # Convert the input image `in` from grayscale to BGR

  out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  # Loop over the clusters
  for i in range(len(clusters)):
      # Generate random colors for each cluster
      color = (i * random.randint(0, 255) % 255, (i + 2) * random.randint(0, 255) % 255, (i + 1) * 100 % 255)
      if i == 0:
          color = (0, 255, 0)  # Green for the first cluster
      elif i == 1:
          color = (0, 0, 255)  # Red for the second cluster
      
      # Draw lines between consecutive points in the cluster
      for j in range(1, len(clusters[i])):
          pt1 = tuple(map(int, clusters[i][j - 1]))  # Ensure points are tuples
          pt2 = tuple(map(int, clusters[i][j]))      # Ensure points are tuples
          cv2.line(out, pt1, pt2, color, 2)

  plt.figure(figsize=(10,6))
  plt.imshow(out)
  plt.show()
            
        
    
     
splitcurve(testFile)