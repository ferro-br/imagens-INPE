import cv2 
import matplotlib.pyplot as plt
from PIL  import Image
from utils import *
cls()

# Load images
print("OpenCV version:", cv2.__version__)
sample_file = './_Amostras/clear1/CP_2016_0803/OH_CP_20160804_083958.tif'
img = cv2.imread(sample_file)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    