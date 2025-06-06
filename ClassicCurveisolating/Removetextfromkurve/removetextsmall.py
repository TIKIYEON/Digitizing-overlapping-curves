import cv2
import pytesseract
import imageio.v3 as iio
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from skimage.filters import threshold_triangle
from skimage.morphology import disk, extrema
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#img = cv2.imread('T14502Las/T14502_02-Feb-07_JewelryLog-Kopi.tiff')
testfolder = "testfolder/chunks/chunk1.tif"
#testfolder = "../testfolder/fulltext.tif"
#cv2.imwrite("testfolder/text.jpg",img)
image = iio.imread(testfolder)
#gray = ski.color.rgb2gray(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = ndimage.gaussian_filter(gray1, sigma=2)
# img = 255 - img #invert image
#text found
text = pytesseract.image_to_string(image)
print(text)
boxes = pytesseract.image_to_boxes(image)
shape = boxes.splitlines()
print(shape)

#mask the  box
mask = np.zeros_like(gray)
changes = []
for b in boxes.splitlines():
  b = b.split(' ')
  s, x, y, w, h = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])
  #If remove lines is needed, change this.
  if s != '~' :
    changes = np.append(changes,b)
    cv2.rectangle(mask, (x, image.shape[0] - y), (w, image.shape[0]-h), 255, -1)

mask2 = mask > 254
image3 = image.copy()
image3[mask2] = [255,255,255]
""" inpainted = cv2.inpaint(image, mask, inpaintRadius=50, flags=cv2.INPAINT_TELEA) """
fig, ax = plt.subplots()
ax.imshow(image3)
print("plot 1:")
plt.show()
image4 = image3.copy()
gray2 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray2, threshold1=50, threshold2=150)

""" # Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) """

""" kernel = np.ones((2, 5), np.uint8)  """

fig, ax = plt.subplots()
ax.imshow(edges, cmap='gray')
print("plot 2:")
plt.show()
print(edges)
print(changes)



