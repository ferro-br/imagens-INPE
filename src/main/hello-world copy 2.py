
import streamlit as st
from utils import *
from keras import models
# Import the new bulk feature extraction function from your new file
from streamlit_utils import extract_features_bulk_for_streamlit_uploads 


st.write("Hello world")

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
CONV_NET = models.load_model(CONV_NET_FILE)

# =====================================================================
st.title("Upload Files From a Folder")

st.write("Please, select the files you want to process.")


# Initialize session state for the uploader key IF IT DOESN'T EXIST
if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0

# --- The File Uploader ---
# This is the crucial part: the 'key' argument for st.file_uploader
# must always be present and use the value from st.session_state.
uploaded_files = st.file_uploader(
    "Drag and drop the pictures here to process them",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state['file_uploader_key']}" # This dynamic key is essential
)

# --- The Reset Button ---
# The button should be placed where it makes sense in your UI flow,
# but its action updates the session state variable that the uploader uses.
if st.button("Reset Uploads"):
    st.session_state['file_uploader_key'] += 1
    st.experimental_rerun() # Force rerun to apply the new key and clear the uploader

if uploaded_files:    
    st.success(f"{len(uploaded_files)} files selected and redy for processing.")
    with st.spinner("Extracting features from images... This might take a moment."):
            processed_images, file_names = extract_features_bulk_for_streamlit_uploads(
                uploaded_files, SAMPLE_DIMS, INTERPOL_METHOD, False, True
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
            for i, pred_prob_array in enumerate(predictions): # pred_prob_val is the single probability for positive class
                pred_prob_val = pred_prob_array.item()                     
                predicted_class_index = 1 if pred_prob_val >= 0.5 else 0
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                st.markdown(f"**Image:** `{file_names[i]}`")
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Probability of Luminescence:** `{pred_prob_val:.4f}`")
                st.write(f" &nbsp;&nbsp;&nbsp;&nbsp;**Predicted Class:** `{predicted_class_name}` (Label: `{predicted_class_index}`)")
                st.write("---")
    else:
        st.warning("No valid images were processed for feature extraction. Please check the uploaded files or console for errors.")

else:
    st.info("Please upload image files (.tif or .tiff) to begin processing.")
    
