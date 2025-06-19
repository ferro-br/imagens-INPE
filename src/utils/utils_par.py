import cv2
import multiprocessing
from functools import partial
from utils import *

# Function to extract features from images
def extract_features(file_name, dims, interpol):
    img = cv2.imread(file_name)
    # Resize the image to a specific size (width, height)
    img_reduced = cv2.resize(img, dims, interpolation = interpol)  # Adjust the size ("dims") as needed

    # Reshape the image into a 1D array
    features = img_reduced.reshape(-1)

    return features

# Extract features, multiples files
def extract_features2(data_sample, dims, interpol):
    image_path = data_sample[0]
    label      = data_sample[1]
    features   = extract_features(image_path, dims, interpol)
    return (features, label)
