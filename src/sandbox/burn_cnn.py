
import cv2
import numpy as np
from utils import *
from utils_img import *

image_path_no_pat    = '..\..\_samples\clear\clear1\CP_2016_0803\OH_CP_20160803_214225.tif'
image_path_pat    = '..\..\_samples\clear\clear2\CP_2016_0408\OH_CP_20160408_215233.tif' # './img/cachorro.tif'
image_path_dark1    = '..\..\_samples\dark\CP_2016_0727\O6_CP_20160727_213914.tif'
image_path_dark2    = '..\..\_samples\dark\CP_2016_0727\O6_CP_20160727_222207.tif'

image_path = image_path_pat

# variance_list: A list of variances to create multiple Gaussian filters with varying degrees of smoothing. 
# Used with Multi-Scale Retinex (MSR) 
variance_list = [24, 128, 48]; # [15, 80, 250] #[15, 80, 30] # Used for MSR

# Variance: controls the spread or width of the Gaussian function. 
# A higher variance results in a wider, smoother filter. Used with Single-Scale Retinex (SSR) 
variance      = 10 #24 #300 # Used for SSR

sample_dims   = (200, 200) # pictures' size (dimensions in pixel) after resizeing (to extract the features)
interpol_method = cv2.INTER_LANCZOS4 # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4
    
start_time = tic()
img_orig = cv2.imread(image_path)
img_msr  = applyRetinex(image_path, sample_dims, interpol_method, 'MSR', variance_list)
img_ssr  = applyRetinex(image_path, sample_dims, interpol_method, 'SSR', variance)

tac(start_time)

cv2.imshow('Original', img_orig)
cv2.imshow('MSR', img_msr)
cv2.imshow('SSR', img_ssr)
#cv2.imwrite('SSR.jpg', img_ssr)
#cv2.imwrite('MSR.jpg',img_msr)

cv2.waitKey(0)
cv2.destroyAllWindows()