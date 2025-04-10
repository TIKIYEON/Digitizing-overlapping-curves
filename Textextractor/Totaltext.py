import cv2
import pytesseract
import imageio.v3 as iio
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_triangle
from skimage.morphology import disk, extrema
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
testfolder = '../T14502Las/T14502_02-Feb-07_JewelryLog-Kopi.tiff'
#testfolder = '../testfolder/stampremoved.tiff'
#testfolder = "testfolder/chunks/chunk1.tif"
#cv2.imwrite("testfolder/text.jpg",img)
image = cv2.imread(testfolder)
#image = np.rot90(img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray, threshold1=50, threshold2=150)
# img = 255 - img #invert image
#text found
text = pytesseract.image_to_string(image)
print(text)
boxes = pytesseract.image_to_boxes(image)


#mask the 
fig, ax = plt.subplots()
ax.imshow(image)
print("plot 1:")
plt.show()
