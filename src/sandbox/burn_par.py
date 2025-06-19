import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import *
from utils_par import *

# Parameters of the simulation
sample_dims     = (50, 50) # pictures' size (dimensions in pixel) after resizeing (to extract the features)
pics_extension  = '.tif' # picture file's extension
clear_pics_path = './_Amostras/clear' # path of the positive pictures
positive_label  = 1 # label of the positive samples
dark_pics_path  = './_Amostras/dark' # path of the negative pictures
negative_label  = 0 # label of the positive samples
perc_test_set   = 0.2 # Percentage of the original dataset that will be used to test the model
interpol_method = cv2.INTER_LANCZOS4 # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4
num_workers     = 8  # for multiprocessing


# Load and preprocess images
samples_files_pos = get_files_by_ext(clear_pics_path, pics_extension) # list of the positive samples
samples_files_neg = get_files_by_ext(dark_pics_path,  pics_extension) # list of the negative samples

# Assembles the whole dataset in a random fashion, with labels
whole_data_set = merge_randomly_with_labels(samples_files_pos, samples_files_neg, positive_label, negative_label) 

sampTot = len(whole_data_set) # amount of samples

############################################################################################################
# Processes the images to extract their features
print("AAAAAAAAAAExtracting features from the data files... ", end='')
start_time = tic()
dataset = []
if __name__ == '__main__':
    # Create a partial function with the constant argument        
    partial_extract_features = partial(extract_features2, dims=sample_dims, interpol=interpol_method)
    with multiprocessing.Pool(processes=num_workers) as pool:
        t = pool.map(partial_extract_features, whole_data_set)
        print(f"length: {len(t)}")
        input("Press something...")
    dataset.extend(t)

input("Press something...")
X, y = zip(*dataset)
X = np.array(X)
y = np.array(y)
print('Done. \n\n\n', end='')
tac(start_time)

############################################################################################################

# Split data into training and testing sets
print("Creating test and training sets... ", end='')
start_time = tic()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=perc_test_set)
tac(start_time)

# Create and train the logistic regression model
print("Creating and training a logistic model... ", end='')
start_time = tic()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print('Done.', end='')
tac(start_time)

input("Done. Push a key to continue...")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'The model reached {accuracy*100}% of accuracy.')