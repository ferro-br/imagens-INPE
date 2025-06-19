"""
     ClearShotIdentifier.py
     This script load a batch of positive and negative samples (i.e., pictures), reduce 
     their resolutions to a standard value, scramble them and use them to train a logistic 
     model. The model should predict if an unknown picture is "positive" or "negative".
     NOTE: Positive images are "usable" photos (clear shots) and negative ones are 
           "unusable" photos (dark shots)
"""
import cv2
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import *

# Parameters of the simulation

MODEL_FILENAME     = 'logistic_regression_model.joblib' # File where the model wil be saved to
SAMPLE_DIMS        = (50, 50) # A tuple pictures' size (dimensions in pixel) after resizeing (to extract the features)
PICS_EXTENSION     = '.tif' # picture file's extension
POSITIVE_PICS_PATH = '../../_Samples/clear' # path of the positive pictures
NEGATIVE_PICS_PATH = '../../_Samples/dark' # path of the negative pictures
POSITIVE_LABEL     = 1 # label of the positive samples (clear shots)
NEGATIVE_LABEL     = 0 # label of the positive samples (dark shots)
PERC_TEST_SET      = 0.2 # Percentage of the original dataset that will be used to test the model
INTERPOL_METHOD    = cv2.INTER_LINEAR # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4
num_workers        = 8  # for multiprocessing

# Gets all the image files
samples_files_pos = get_files_by_ext(POSITIVE_PICS_PATH, PICS_EXTENSION) # list of the positive samples
samples_files_neg = get_files_by_ext(NEGATIVE_PICS_PATH, PICS_EXTENSION) # list of the negative samples

# Assembles the whole dataset in a random fashion, with labels
whole_data_set = merge_randomly_with_labels(samples_files_pos, samples_files_neg, POSITIVE_LABEL, NEGATIVE_LABEL) 

# Extract the features of the samples
print(f"Going to extract features by using {INTERPOL_METHOD}...")
X, y, samp, samp1, samp0, _ = extract_features_bulk(whole_data_set, SAMPLE_DIMS, INTERPOL_METHOD, True, True)
print(f"{samp0} samples have label 0 and {samp1} samples have label 1.")

# Split data into training and testing sets
print("Creating test and training sets... ", end='')
start_time = tic()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PERC_TEST_SET)
tac(start_time)

# Create and train the logistic regression model
print("Creating and training a logistic model... ", end='')
start_time = tic()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print('Done.')

print('Saving the model on disk... ', end='')

# Save model on disk. To retrieve it: model = joblib.load(MODEL_FILENAME)
joblib.dump(model, MODEL_FILENAME)

print('Done.', end='')
tac(start_time)

#input("Done. Push a key to continue...")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'The model reached {accuracy*100}% of accuracy.')