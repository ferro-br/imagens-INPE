import streamlit as st
from utils import *
from keras import models

st.write("Hello world")

# Name of the file on disk who will store the CNN created 
CONV_NET_FILE = 'luminiscence_cnn_model.keras'

POSITIVE_LABEL = 1 # Label for "contains luminescence patterns"
NEGATIVE_LABEL = 0 # Label for "does not contain luminescence patterns"
CLASS_NAMES    = ['Luminescences detected', 'No luminescences detected'] 
PATH_FILES     =  '..\..\_Samples' # Location of the files (can contain subfolders)

# Parameters of the features
SAMPLE_DIMS     = (50, 50) # A tuple to hold the pictures' size (dimensions in pixels) after resizeing (to extract the features)
PERC_TEST_SET   = 0.2 # Percentage of the original dataset that will be used to test the model
INTERPOL_METHOD = cv2.INTER_LINEAR # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4

# Gets all the image files
LIST_FILES = get_files_by_ext(PATH_FILES, 'tif') # files with positive samples

# Extract the features of the samples (each "sample" is a file name of a picture)
processed_images, file_names = extract_features_bulk2(LIST_FILES, SAMPLE_DIMS, INTERPOL_METHOD, False, True)

# read a convolutional NN ftom disk
CONV_NET = models.load_model(CONV_NET_FILE)

# Makes the CNN work, i.e., make predictions
predictions = CONV_NET.predict(processed_images)

# Interpret the predictions
# Since the output layer of "CONV_NET" has 1 neuron with 'sigmoid' activation,
# the prediction will be a probability between 0 and 1.
# You'll likely set a threshold (e.g., 0.5) to classify.

st.write("\n--- Predictions ---")
for i, pred_prob in enumerate(predictions):
    predicted_class_index = (pred_prob > 0.5).astype(int)[0] # Threshold at 0.5
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    st.write(f"Image: {file_names[i]}")
    st.write(f"  Predicted Probability: {pred_prob[0]:.4f}")
    st.write(f"  Predicted Class: {predicted_class_name} (Label: {predicted_class_index})")
    st.write("-" * 20)