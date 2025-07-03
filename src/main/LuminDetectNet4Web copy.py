import sys
import os

# --- FORCE REPOSITORY ROOT INTO SYSPATH (FOR LOCAL DEVELOPMENT ONLY) ---
# Get the absolute path to the directory containing this script (src/main)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach the repository root (e.g., D:\...\Luminiscencias)
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning to prioritize it

import streamlit as st
from src.streamlit_utils.streamlit_utils import *
from keras import models
import cv2
import joblib

st.set_page_config(layout="wide")

st.write("Hello world") # Consider removing or integrating this later

# Name of the file on disk which stores the CNN created 
CONV_NET_FILE   = 'luminiscence_cnn_model.keras'

# Name of the file on disk that tells "usable" pictures (clear shots) from "unusable" ones (dark shots)
LOGISTIC_FILE   = 'logistic_regression_model.joblib' 

POSITIVE_LABEL  = 1 # Label for "contains luminescence patterns"
NEGATIVE_LABEL  = 0 # Label for "does not contain luminescence patterns"
CNN_CLASS_NAMES = ['Waves Patterns detected', 'Waves Patterns detected'] 
LOG_CLASS_NAMES = ['Clear Shot (Usable)', 'Dark Shot (Unusable)'] # Names for the logistic classifier's classes


# Parameters of the features
SAMPLE_DIMS     = (50, 50) # A tuple to hold the pictures' size (dimensions in pixels) after resizeing (to extract the features)
PERC_TEST_SET   = 0.2 # Percentage of the original dataset that will be used to test the model
INTERPOL_METHOD = cv2.INTER_LINEAR # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4

# Reads a convolutional NN from disk
try:
    CONV_NET = models.load_model(CONV_NET_FILE)
except Exception as e:
    st.error(f"Error loading the CNN model: {e}. Please ensure '{CONV_NET_FILE}' is in the correct directory.")
    st.stop()

# Reads a logistic classifier from disk
try:
    LOG_CLASSIFIER = joblib.load(LOGISTIC_FILE)
except Exception as e:
    st.error(f"Error loading the Logistic Regression model: {e}. Please ensure '{LOGISTIC_FILE}' is in the correct directory.")
    st.stop() # Stop the app if the model can't be loaded

# =====================================================================
st.title("Upload Files From a Folder")

st.write("Please, select the files you want to process.")

# --- Session State Initialization for Uploader Key and Stored Files ---
if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0
if 'uploaded_files_data' not in st.session_state:
    st.session_state['uploaded_files_data'] = None
if 'processing_results' not in st.session_state:
    st.session_state['processing_results'] = []  # Initialize processing_results

# --- The File Uploader ---
uploaded_files = st.file_uploader(
    "Drag and drop the pictures here to process them",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state['file_uploader_key']}" # This dynamic key is essential
)

# Store uploaded files in session state only when new files are uploaded
# This prevents them from being "lost" on a rerun that isn't a "Process" button click
if uploaded_files:
    # Only update if the files are truly new or different from what's stored
    # This check prevents re-storing the same files on every rerun
    if uploaded_files != st.session_state['uploaded_files_data']:
        st.session_state['uploaded_files_data'] = uploaded_files
        st.success(f"{len(uploaded_files)} files selected and ready for processing. Click 'Process Files' to begin.")
else:
    # If the uploader is empty (e.g., after reset), clear the session state storage too
    st.session_state['uploaded_files_data'] = None
    st.info("Please upload image files (.tif or .tiff) to begin processing.")


# --- Process Files Button ---
# This button will trigger the feature extraction and prediction
process_button = st.button("Process Files")

# --- The Reset Button ---
if st.button("Reset Uploads"):
    st.session_state['file_uploader_key'] += 1 # Increment key to reset uploader
    st.session_state['uploaded_files_data'] = None # Clear stored files
    st.session_state['processing_results'] = []  # Initialize processing_results
    st.rerun() # Force rerun to apply the new key and clear data

# --- Conditional Processing Logic ---
# This block now only runs when the "Process Files" button is clicked AND files are available
if process_button and st.session_state['uploaded_files_data']:
    st.write("--- Starting Processing ---") # Added for clarity
    with st.spinner("Extracting features from images... This might take a moment."):
        processed_images, file_names, reduced_images, original_images = extract_features_bulk_for_streamlit_uploads(
            st.session_state['uploaded_files_data'], SAMPLE_DIMS, INTERPOL_METHOD, True)
    
    if processed_images.size > 0: # Check if any images were processed
        st.success(f"{len(file_names)} images processed and features extracted.")
        # --- Make predictions with Logistic Classifier ---
        st.write("--- Applying Clear/Dark Shot Classification ---")
        with st.spinner("Classifying images as clear or dark shots..."):
            features_for_logistic = processed_images.reshape(processed_images.shape[0], -1)
            log_predictions = LOG_CLASSIFIER.predict(features_for_logistic)             
            st.success("Logistic classification completed!")

        # --- Make predictions with CNN (Luminescence) Classifier ---
        st.write("--- Applying Luminescence Detection (CNN) ---")
        with st.spinner("Classifying images for luminescence patterns..."):
            # The CNN uses the original 4D processed_images, as that's its expected input shape.
            cnn_predictions = CONV_NET.predict(processed_images) 
            st.success("CNN Luminescence detection completed!")
            
        st.write("\n--- Combined Prediction Results ---")
        for i in range(len(file_names)):
            file_name = file_names[i]
            
            # Get Logistic Regression result
            log_pred_value      = log_predictions[i]
            log_pred_class      = LOG_CLASS_NAMES[1-log_pred_value]

            # Get CNN result
            cnn_pred_prob_array = cnn_predictions[i]                     
            cnn_pred_prob_val   = cnn_pred_prob_array.item()                     
            cnn_pred_value      = 1 if cnn_pred_prob_val >= 0.5 else 0
            cnn_pred_class      = CNN_CLASS_NAMES[1-cnn_pred_value]
            
            # Store results
            st.session_state['processing_results'].append({
                "file_name": file_name,           # Name of the file
                "log_pred_class": log_pred_class, # Class of the picture (clear or dark), given by the logistic model
                "log_pred_value": log_pred_value, # Output value of the logistic model
                "cnn_pred_prob_val": f"{cnn_pred_prob_val:.4f}", # Output value of the Conv Net model
                "cnn_pred_class": cnn_pred_class, # Class of the Conv Net model (with/witout luminescences)
                "cnn_pred_value": cnn_pred_value,  # Output value of the Conv Net model (rounded)
                "image_data": original_images[i] # Store the original image here                
                # "reduced_image_data": reduced_images[i]                
            })
            
            # Display results
            st.markdown(f"**Image:** `{file_name}`")
            st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Initial Classification (Clear/Dark):** `{log_pred_class}`")
            st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Luminescence Probability:** `{cnn_pred_prob_val:.4f}`")
            if log_pred_value == 1:
               st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Luminescence Predicted Class:** `{cnn_pred_class}` (Label: `{cnn_pred_value}`)")
            else:
               st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Luminescence Predicted Class:** `{cnn_pred_class}` (Irrelevant since the image is unusable)")
            st.write("---")
        
        st.success("Overall processing complete! Results are displayed above and ready for PDF report.")

    else:
        st.warning("No valid images were processed for feature extraction. Please check the uploaded files or console for errors.")
        st.session_state['processing_results'] = None 
elif process_button and not st.session_state['uploaded_files_data']:
    st.warning("Please upload files before clicking 'Process Files'.")

# --- PDF Generation Logic ---
if st.session_state['processing_results']:
    st.write("---")
    st.header("Generate Report")

    # Generate the PDF
    pdf_buffer = create_pdf_report(st.session_state['processing_results'])

    # Add a download button for the PDF
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="luminescence_report.pdf",
        mime="application/pdf"
    )
