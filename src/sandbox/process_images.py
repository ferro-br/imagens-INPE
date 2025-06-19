import cv2 
import matplotlib.pyplot as plt
from PIL  import Image
from utils import *
cls()

# Load images
samples_files = get_files_by_ext('./_Amostras/clear1', '.tif')
print("OpenCV version:", cv2.__version__)

for sample_file in samples_files:
    print(f"Exhibiting {sample_file}...")
    img = cv2.imread(sample_file)
    cv2.imshow('Image', img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cv2.destroyAllWindows()    

