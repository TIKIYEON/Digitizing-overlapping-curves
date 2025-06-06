import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#img = cv2.imread('T14502Las/T14502_02-Feb-07_JewelryLog-Kopi.tiff')
#testfolder = "testfolder"
#cv2.imwrite("testfolder/text.jpg",img)
temp = cv2.imread("text.jpg")
# img = 255 - img #invert image
print(pytesseract.image_to_string(temp))

print('====================================================')