import sys
import os
import numpy as np
import io # NEW: For in-memory file operations
import zipfile # NEW: For creating zip archives

# --- FORCE REPOSITORY ROOT INTO SYSPATH (FOR LOCAL DEVELOPMENT ONLY) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.streamlit_utils.streamlit_utils import *
from keras import models
import cv2
import joblib

st.set_page_config(layout="wide")

st.write("Hello world") # Consider removing or integrating this later

# Name of the file on disk which stores the CNN created
CONV_NET_FILE = 'luminiscence_cnn_model.keras'

# Name of the file on disk that tells "usable" pictures (clear shots) from "unusable" ones (dark shots)
LOGISTIC_FILE = 'logistic_regression_model.joblib'

POSITIVE_LABEL = 1 # Label for "contains waves patterns"
NEGATIVE_LABEL = 0 # Label for "does not contain waves patterns"
CNN_CLASS_NAMES = ['Waves patterns detected', 'Waves patterns NOT detected']
LOG_CLASS_NAMES = ['Clear Shot (Usable)', 'Dark Shot (Unusable)'] # Names for the logistic classifier's classes


# Parameters of the features
SAMPLE_DIMS = (50, 50) # A tuple to hold the pictures' size (dimensions in pixels) after resizeing (to extract the features)
PERC_TEST_SET = 0.2 # Percentage of the original dataset that will be used to test the model
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
    st.session_state['processing_results'] = [] # Initialize processing_results
# NEW: Session state for the generated zip file
if 'classified_zip_buffer' not in st.session_state:
    st.session_state['classified_zip_buffer'] = None

# --- The File Uploader ---
uploaded_files = st.file_uploader(
    "Drag and drop the pictures here to process them",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state['file_uploader_key']}" # This dynamic key is essential
)

# Store uploaded files in session state only when new files are uploaded
if uploaded_files:
    if uploaded_files != st.session_state['uploaded_files_data']:
        st.session_state['uploaded_files_data'] = uploaded_files
        # Clear previous zip buffer if new files are uploaded
        st.session_state['classified_zip_buffer'] = None 
        st.success(f"{len(uploaded_files)} files selected and ready for processing. Click 'Process Files' to begin.")
else:
    st.session_state['uploaded_files_data'] = None
    st.session_state['classified_zip_buffer'] = None # Clear zip buffer if uploader is empty
    st.info("Please upload image files (.tif or .tiff) to begin processing.")


# --- Process Files Button ---
process_button = st.button("Process Files")

# --- The Reset Button ---
if st.button("Reset Uploads"):
    st.session_state['file_uploader_key'] += 1 # Increment key to reset uploader
    st.session_state['uploaded_files_data'] = None # Clear stored files
    st.session_state['processing_results'] = [] # Initialize processing_results
    st.session_state['classified_zip_buffer'] = None # Clear zip buffer on reset
    st.rerun() # Force rerun to apply the new key and clear data

# --- Conditional Processing Logic ---
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
            ind_clear_shots = np.where(log_predictions == 1)[0]
            st.success("Logistic classification completed!") 

        clear_shot_images = processed_images[ind_clear_shots]
        
        cnn_predictions = np.array([]) 

        if clear_shot_images.size > 0:
            st.write("--- Applying Wave Patterns Detection (CNN) ---")
            with st.spinner("Classifying images for waves patterns..."):
                cnn_predictions = CONV_NET.predict(clear_shot_images) 
                st.success("Search for waves patterns completed!")
        else:
            st.info("No clear shots detected. Skipping wave pattern classification.")
        
        st.write("\n--- Combined Prediction Results ---")
        contImgCNN = 0

        # Clear previous results before adding new ones
        st.session_state['processing_results'] = [] 
        st.session_state['classified_zip_buffer'] = None # Ensure zip buffer is cleared on new processing
        
        for contImg in range(len(file_names)): # iterates over all uploaded images
            file_name = file_names[contImg]
            
            log_pred_value = log_predictions[contImg]
            log_pred_class = LOG_CLASS_NAMES[1 - log_pred_value]

            cnn_pred_prob_val = -1.0
            cnn_pred_value = -1
            cnn_pred_class = 'Not classified because the shot was not clear'

            if log_pred_value == 1:
                if contImgCNN < len(cnn_predictions):
                    cnn_pred_prob_array = cnn_predictions[contImgCNN]
                    cnn_pred_prob_val = cnn_pred_prob_array.item()
                    cnn_pred_value = 1 if cnn_pred_prob_val >= 0.5 else 0
                    cnn_pred_class = CNN_CLASS_NAMES[1 - cnn_pred_value]
                    contImgCNN += 1
                else:
                    st.warning(f"Internal error: Mismatch in CNN predictions count for {file_name}. Assigning default.")

            st.session_state['processing_results'].append({
                "file_name": file_name,
                "log_pred_class": log_pred_class,
                "log_pred_value": log_pred_value,
                "cnn_pred_prob_val": cnn_pred_prob_val,
                "cnn_pred_class": cnn_pred_class,
                "cnn_pred_value": cnn_pred_value,
                "image_data": original_images[contImg]
            })
            
            st.markdown(f"**Image:** `{file_name}`")
            st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Image quality classification (Clear/Dark):** `{log_pred_class}`")
            if log_pred_value == 1:
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Wave patterns detected with probability:** `{cnn_pred_prob_val:.4f}`")
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Predicted Class:** `{cnn_pred_class}` (Label: `{cnn_pred_value}`)")
            else:
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Wave patterns detected with probability:** `N/A` (Image was dark/unusable)")
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Predicted Class:** `{cnn_pred_class}`")
            st.write("---")
        
        st.success("Overall processing complete! Results are displayed above and ready for PDF report and image saving.")

    else:
        st.warning("No valid images were processed for feature extraction. Please check the uploaded files or console for errors.")
        st.session_state['processing_results'] = []
elif process_button and not st.session_state['uploaded_files_data']:
    st.warning("Please upload files before clicking 'Process Files'.")


# --- Function to create zip file in memory ---
# Folder names to be used inside the zip archive
ZIP_BASE_FOLDER_NAME = "Classified_Images_Report"
ZIP_DARK_SHOTS_FOLDER = os.path.join(ZIP_BASE_FOLDER_NAME, "Dark_Shots")
# Use the actual class names for subfolders in Clear_Shots, replacing spaces
ZIP_WAVES_DETECTED_FOLDER = os.path.join(ZIP_BASE_FOLDER_NAME, "Clear_Shots", CNN_CLASS_NAMES[0].replace(" ", "_"))
ZIP_WAVES_NOT_DETECTED_FOLDER = os.path.join(ZIP_BASE_FOLDER_NAME, "Clear_Shots", CNN_CLASS_NAMES[1].replace(" ", "_"))


def create_classified_images_zip():
    if not st.session_state['processing_results']:
        st.warning("No processing results available to create a zip file.")
        return None # Return None if no results

    zip_buffer = io.BytesIO()
    
    # Create the zip file in the in-memory buffer
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zipf:
        added_count = 0
        for result in st.session_state['processing_results']:
            file_name = result["file_name"]
            image_data = result["image_data"]

            if image_data is None or not isinstance(image_data, np.ndarray):
                st.warning(f"Skipping '{file_name}' for zip: Image data is missing or not a valid NumPy array.")
                continue

            log_pred_value = result["log_pred_value"]
            cnn_pred_value = result["cnn_pred_value"]

            # Determine the path where this file should go inside the zip
            arcname = "" # Archive Name (path inside the zip)
            if log_pred_value == 0: # Dark Shot
                arcname = os.path.join(ZIP_DARK_SHOTS_FOLDER, file_name)
            elif log_pred_value == 1: # Clear Shot
                if cnn_pred_value == 1: # Waves Patterns detected
                    arcname = os.path.join(ZIP_WAVES_DETECTED_FOLDER, file_name)
                elif cnn_pred_value == 0: # Waves Patterns NOT detected
                    arcname = os.path.join(ZIP_WAVES_NOT_DETECTED_FOLDER, file_name)
                else: # Fallback for unexpected CNN value for a clear shot
                    st.warning(f"Skipping '{file_name}' for zip: Clear shot but CNN prediction was inconclusive ({cnn_pred_value}).")
                    continue
            else: # Fallback for unexpected logistic value
                st.warning(f"Skipping '{file_name}' for zip: Unexpected logistic prediction value ({log_pred_value}).")
                continue

            try:
                # Get the file extension (e.g., '.tif', '.tiff')
                file_ext = "." + file_name.split('.')[-1].lower()
                # Encode the image data into bytes
                is_success, encoded_image = cv2.imencode(file_ext, image_data)
                
                if is_success:
                    zipf.writestr(arcname, encoded_image.tobytes())
                    added_count += 1
                else:
                    st.error(f"Failed to encode image '{file_name}' for zip. Check file format support.")
            except Exception as e:
                st.error(f"Failed to add '{file_name}' to zip: {e}")
                
    st.success(f"Prepared {added_count} images for zip archive.")
    zip_buffer.seek(0) # Rewind the buffer to the beginning
    return zip_buffer


# --- PDF Generation and Image Saving Buttons ---
if st.session_state['processing_results']:
    st.write("---")
    st.header("Generate Report and Download Classified Images")

    # Add a download button for the PDF
    pdf_buffer = create_pdf_report(st.session_state['processing_results'])
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="luminescence_report.pdf",
        mime="application/pdf"
    )

    # NEW: Button to trigger ZIP creation
    # This button prepares the zip file and stores it in session state
    if st.button("Generate Classified Images (ZIP)"):
        with st.spinner("Creating zip archive... This may take a moment for large datasets."):
            st.session_state['classified_zip_buffer'] = create_classified_images_zip()
        if st.session_state['classified_zip_buffer']:
            st.success("Zip archive created! Click the button below to download.")
        else:
            st.error("Failed to create zip archive.")
    
    # NEW: Download button for the ZIP (appears only if the zip buffer exists)
    if st.session_state['classified_zip_buffer']:
        st.download_button(
            label="Download Classified Images (ZIP)",
            data=st.session_state['classified_zip_buffer'],
            file_name="classified_images.zip",
            mime="application/zip",
            key="download_classified_zip_button" # Unique key to prevent issues with other buttons
        )