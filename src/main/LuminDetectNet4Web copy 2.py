
import streamlit as st
from utils import *
from keras import models
# Import the new bulk feature extraction function from your new file
from streamlit_utils import *
import io # To handle the PDF in memory
from fpdf import FPDF # Import FPDF

st.set_page_config(layout="wide")

st.write("Hello world") # Consider removing or integrating this later

# Name of the file on disk who will store the CNN created 
CONV_NET_FILE = 'luminiscence_cnn_model.keras'

POSITIVE_LABEL = 1 # Label for "contains luminescence patterns"
NEGATIVE_LABEL = 0 # Label for "does not contain luminescence patterns"
CLASS_NAMES    = ['Luminescences detected', 'No luminescences detected'] 

# Parameters of the features
SAMPLE_DIMS     = (50, 50) # A tuple to hold the pictures' size (dimensions in pixels) after resizeing (to extract the features)
PERC_TEST_SET   = 0.2 # Percentage of the original dataset that will be used to test the model
INTERPOL_METHOD = cv2.INTER_LINEAR # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4

# read a convolutional NN ftom disk
try:
    CONV_NET = models.load_model(CONV_NET_FILE)
except Exception as e:
    st.error(f"Error loading the CNN model: {e}. Please ensure '{CONV_NET_FILE}' is in the correct directory.")
    st.stop()

# =====================================================================
st.title("Upload Files From a Folder")

st.write("Please, select the files you want to process.")

# --- Session State Initialization for Uploader Key and Stored Files ---
if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0
if 'uploaded_files_data' not in st.session_state:
    st.session_state['uploaded_files_data'] = None
if 'processing_results' not in st.session_state:
    st.session_state['processing_results'] = None  # Initialize processing_results

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
    st.rerun() # Force rerun to apply the new key and clear data


# --- Conditional Processing Logic ---
# This block now only runs when the "Process Files" button is clicked AND files are available
if process_button and st.session_state['uploaded_files_data']:
    st.write("--- Starting Processing ---") # Added for clarity
    with st.spinner("Extracting features from images... This might take a moment."):
        processed_images, file_names = extract_features_bulk_for_streamlit_uploads(
            st.session_state['uploaded_files_data'], SAMPLE_DIMS, INTERPOL_METHOD, False, True
        )
    
    if processed_images.size > 0: # Check if any images were processed
        st.success(f"{len(file_names)} images processed and features extracted.")
        # Make predictions
        st.write("--- Making Predictions ---")
        with st.spinner("Classifying images..."):
            predictions = CONV_NET.predict(processed_images)
            st.success("Predictions completed!")
            # Interpret and display predictions
            st.write("\n--- Prediction Results ---")
            for i, pred_prob_array in enumerate(predictions):
                pred_prob_val = pred_prob_array.item()                     
                predicted_class_index = 1 if pred_prob_val >= 0.5 else 0
                predicted_class_name = CLASS_NAMES[1-predicted_class_index]
                st.markdown(f"**Image:** `{file_names[i]}`")
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Probability of Luminescence:** `{pred_prob_val:.4f}`")
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Predicted Class:** `{predicted_class_name}` (Label: `{predicted_class_index}`)")
                st.write("---")
    else:
        st.warning("No valid images were processed for feature extraction. Please check the uploaded files or console for errors.")
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
