import cv2
import numpy as np
from PIL import Image
from retinex import *

def applyRetinex(image_path, sample_dims, interpol_method, processing_method, variance):
    img     = cv2.imread(image_path)
    img     = cv2.resize(img, sample_dims, interpolation = interpol_method)  # Adjust the size ("dims") as needed
    img_ret = None
    if (processing_method.upper()=='MSR'): # Multi-Scale Retinex (MSR), variance is a list with 3 values like [15, 80, 250]        
        img_ret=MSR(img,variance)
    elif (processing_method.upper()=='SSR'): # Single Scale Retinex (SSR), variance is an integer like 100
        img_ret=SSR(img, variance)       
    return img_ret


def tiff2png(tiff_file, png_file):
    # Converts a TIFF image to PNG.
    with Image.open(tiff_file) as img:
        img.save(png_file)

def tiff2jpg(tiff_file, jpg_file):
  # Converts a TIFF image to a JPG.
  #with Image.open(tiff_file) as img:
  #  img.save(jpg_file, 'JPEG')
  img = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
  cv2.imwrite(jpg_file, img)

def create_dataset(image_paths, labels):
    """
    Creates a dataset in MNIST format from a list of image paths and labels.

    Args:
        image_paths: A list of image paths.
        labels: A list of corresponding labels.

    Returns:
        A tuple of NumPy arrays (X, y) in the MNIST format (a dictionary)
        - X: A 2D NumPy array containing the flattened images.
        - y: A 1D NumPy array containing the labels.

    Usage:
        # Assuming you have a list of image paths and labels
        image_paths = ["image1.jpg", "image2.jpg", ...]
        labels = [0, 1, 0, ...]
        X, y = create_dataset(image_paths, labels)
    """

    X = []
    y = []

    for image_path, label in zip(image_paths, labels):
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image
        img = cv2.resize(img, (28, 28))

        # Flatten the image
        img_flattened = img.reshape(-1)

        X.append(img_flattened)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    dic = {'data': X, 'target': y}
    return dic