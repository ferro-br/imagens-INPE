
import time
import os
import random
import cv2
import numpy as np

def tic():
    start_time = time.time()
    return start_time

def tac(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')   

def get_files(folder_path):
  # Gets a list of all files in a given folder.
  files = []
  for root, dirs, filenames in os.walk(folder_path):
    for filename in filenames:
      files.append(os.path.join(root, filename))
  return files

def get_files_by_ext(folder_path, extension):
  ''' 
  Lists all files with the given extension in a folder.
  Usage example: 
    folder_path = "path/to/your/folder"
    extension = ".txt"
    file_list = get_files_by_ext(folder_path, extension)
  '''
  file_list = []
  for root, _, files in os.walk(folder_path):
    for file in files:
      if file.endswith(extension):
        file_path = os.path.join(root, file)
        file_list.append(file_path)
  return file_list

def merge_randomly(list1, list2):
  """Merges two lists randomly.

  Args:
    list1: The first list.
    list2: The second list.

  Returns:
    A new list containing the elements of both lists, merged randomly.
  """

  merged_list = list1 + list2
  random.shuffle(merged_list)
  return merged_list

def merge_randomly_with_labels(list0, list1, lbl0, lbl1):
  """Merges two lists randomly and give them labels.
     Useful to create datasets for ML in two-classes 
     classification tasks
  Args:
    list0: The first list.
    list1: The second list.
    lbl0: Label for the samples in list0
    lbl1: Label for the samples in list1

  Returns:
    A list of tuples, where each tuple contains an element (i.e., a "sample") and a label.
  """
  merged_list = [(item, lbl0) for item in list0] + [(item, lbl1) for item in list1]
  random.shuffle(merged_list)
  return merged_list


def extract_features2(file_name, dims, interpol_method, reshape=True):
    img = cv2.imread(file_name)
    if img is None:
        print(f"Warning: Could not load image {file_name}. Skipping.")
        return None

    # Convert to grayscale if it's a color image
    if len(img.shape) == 3 and img.shape[2] == 3: # Check if it's BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4: # Check if it's BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # If it's already grayscale (len(img.shape) == 2), no conversion needed

    features = cv2.resize(img, dims, interpolation = interpol_method)

    if reshape:
        features = features.reshape(-1) # For grayscale, this is 50*50 = 2500 features

    return features

def extract_features(file_name, dims, interpol_method, reshape=True):
    # TO DO/CAVEAT: Be aware that it calls cv2.resize on img directly. If 
    # the input images vary (e.g., some grayscale, some color, some RGBA with alpha 
    # channel), cv2.resize might behave differently or result in varying feature 
    # vector lengths if you don't standardize the number of channels first.
    # If you want to use color information, then ensure all images are consistently 
    # 3 channels (e.g., convert grayscale to BGR, strip alpha from RGBA) before 
    # resizing, and reshape(-1) will then flatten 50*50*3 features. The current code 
    # will flatten 3 channels if the image has them. Just make sure it's consistent.

    # file_name contains the file name of the picture
    # dims is a tuple (width, height) with the final dimension of each picture
    # interpol_method controls the interpolation method used to resize the image
    # reshape controls wheter the features should be reshaped to a 1D array or not
    img = cv2.imread(file_name)
    # Resize the image to a specific size (width, height)
    features = cv2.resize(img, dims, interpolation = interpol_method)  # Adjust the size ("dims") as needed

    if reshape:
        # Reshape the image into a 1D array
        features = features.reshape(-1)

    return features

def extract_features_bulk2(data_set, sample_dims, interpol_method, reshape=True, verbose=True):
  # whole_data_set is a list of file names of the samples (pictures)
  # sample_dims is a tuple (width, height) with the final dimension of each picture
  # interpol_method controls the interpolation method used to resize the image
  # reshape controls wheter the features should be reshaped to a 1D array or not
  X          = [] # list to store the samples
  file_names = [] # list to store the file names corresponding to the samples
  samp       = 0 # amount of samples
  start_time_over = tic()
  for data_sample in data_set: # iterates over the list of tuples
    samp+=1
    image_path = data_sample
    if verbose:
      print(f'Processing data sample #{samp}. File {os.path.basename(image_path)} ...', end='')
      start_time = tic()
    features = extract_features(image_path, sample_dims, interpol_method, reshape)
    if verbose: 
      print('Done.')
      tac(start_time)
    X.append(features) # stores the features of the samples (a low resolution copy of the samples)
    file_names.append(image_path) # store the complete file name of each sample
  
  if verbose: 
    tac(start_time_over)
    print(f"\n\nProcessing is done. {samp} files were processed.", end='')

  X = np.array(X)
  return X, file_names


def extract_features_bulk(whole_data_set, sample_dims, interpol_method, reshape=True, verbose=True):
    # whole_data_set is a list of tuples, where each tuple contains a sample (i.e., the file 
    # name of a picture) and its label
    # sample_dims is a tuple (width, height) with the final dimension of each picture
    # interpol_method controls the interpolation method used to resize the image
    # reshape controls wheter the features should be reshaped to a 1D array or not
    # verbose controls wheter the function should output messages or not
  # NOTE: Positive labels are "1", negatives are "0"!!!
  X          = [] # list to store the samples
  y          = [] # list to store the labels
  file_names = [] # list to store the file names corresponding to the samples
  samp  = 0 # amount of samples
  samp1 = 0 # amount of positive samples
  samp0 = 0 # amount of negative samples
  start_time_over = tic()
  for data_sample in whole_data_set: # iterates over the list of tuples
    samp+=1
    image_path = data_sample[0]
    label      = data_sample[1]
    if (label==0):
        samp0+=1
    elif (label==1):
        samp1+=1
    if verbose:
      print(f'Processing data sample #{samp}. File {os.path.basename(image_path)}, label {label}...', end='')
      start_time = tic()
    features = extract_features(image_path, sample_dims, interpol_method, reshape)
    if verbose: 
      print('Done.')
      tac(start_time)
    X.append(features) # stores the features of the samples (a low resolution copy of the samples)
    y.append(label)    # stores the label of each sample
    file_names.append(image_path) # store the complete file name of each sample
  
  if verbose: 
    tac(start_time_over)
    print(f"\n\nProcessing is done. {samp} files were processed.", end='')

  X = np.array(X)
  y = np.array(y)
  return X, y, samp, samp1, samp0, file_names
