import cv2
from utils import *
import numpy as np
from PIL import Image
import io
import time
from fpdf import FPDF
import pandas as pd # Assuming you use pandas for results_df as well
import streamlit as st # Import st if you want to use @st.cache_data here

from keras import models # To load your CNN model
from datetime import datetime # Import datetime
import pytz # Import pytz for timezone aware current time 


def extract_features_from_pil(image_pil: Image.Image, dims: tuple, interpol_method: int, reshape: bool = True) -> np.ndarray:
    """
    Extracts features from a PIL.Image.Image object.

    Args:
        image_pil (PIL.Image.Image): The PIL image object to be processed.
        dims (tuple): A tuple (width, height) with the desired final dimension for the image.
        interpol_method (int): The OpenCV interpolation method (e.g., cv2.INTER_LINEAR).
        reshape (bool): If True, features will be reshaped to a 1D array.
                        If False, it will maintain the 2D/3D dimension of the resized image.

    Returns:
        np.ndarray: The extracted image features as a NumPy array.
                    The shape will be (dims[1], dims[0], channels) or (dims[1] * dims[0] * channels,) if reshape=True.
    """
    if not isinstance(image_pil, Image.Image):
        raise TypeError(f"Expected a PIL.Image.Image object for 'image_pil', but got {type(image_pil)}")

    # Convert the PIL Image object to a NumPy array
    # Handle different image modes for compatibility with OpenCV (expects BGR or Grayscale)
    if image_pil.mode == 'RGBA':
        # Convert RGBA to RGB and then to BGR (which is OpenCV's default for colors)
        img_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGBA2RGB)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif image_pil.mode == 'L': # Grayscale
        # Convert grayscale image to BGR (3 channels) if the CNN expects it
        # Or simply use np.array(image_pil) if the CNN can handle 1 channel
        img_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_GRAY2BGR)
    else: # Assume RGB/BGR or other modes that cv2.resize can handle directly
        img_np = np.array(image_pil)
        if len(img_np.shape) == 2: # If the image is 2D (grayscale) and wasn't converted before
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR) # Convert to 3 channels for consistency

    # Resize the image to the specified size (width, height)
    features = cv2.resize(img_np, dims, interpolation=interpol_method)

    # **Important**: Add normalization here if your CNN expects values between 0-1
    # Example: features = features.astype('float32') / 255.0
    # Or if your CNN uses ImageDataGenerator or similar, it might handle this.

    if reshape:
        # Reshape the features to a 1D array, if requested
        features = features.reshape(-1)

    return features

@st.cache_data # Apply the cache decorator HERE!
def extract_features_bulk_for_streamlit_uploads(
    uploaded_files_list: list,
    sample_dims: tuple,
    interpol_method: int,
    reshape: bool = True,
    verbose: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Processes a list of Streamlit UploadedFile objects, extracts features,
    and returns a NumPy array of features and a list of file names.
    """
    X = []  # list to store the samples (features)
    file_names = []  # list to store the file names corresponding to the samples
    samp = 0  # sample counter

    # Using the tic() and tac() that you defined in utils.py
    start_time_over = time.time()
    if verbose:
        st.info("Starting feature extraction from uploaded images...")

    for uploaded_file_obj in uploaded_files_list:
        samp += 1
        current_file_name = uploaded_file_obj.name

        if verbose:
            print(f'Processing data sample #{samp}. File {current_file_name} ...', end='')
            start_time = time.time()

        try:
            # 1. Read the file content in bytes
            bytes_data = uploaded_file_obj.read()
            # 2. Open the bytes as a PIL Image (Pillow)
            image_pil = Image.open(io.BytesIO(bytes_data))

            # 3. Call the NEW extract_features_from_pil function
            features = extract_features_from_pil(image_pil, sample_dims, interpol_method, reshape)

            X.append(features)
            file_names.append(current_file_name)

            if verbose:
                print('Done.')
                print(f"Elapsed time for {current_file_name}: {time.time() - start_time:.4f} seconds")

        except Exception as e:
            # Catch errors for individual image processing
            st.error(f"Error processing '{current_file_name}': {e}. Skipping this file.")
            print(f"Error processing '{current_file_name}': {e}") # Log to console
            continue  # Skip to the next file

    if verbose:
        total_elapsed_time = time.time() - start_time_over
        st.success(f"Overall processing done. {samp} files attempted, {len(X)} processed successfully in {total_elapsed_time:.2f} seconds.")
        print(f"\n\nOverall processing done. {samp} files attempted, {len(X)} processed successfully.")
        print(f"Total elapsed time: {total_elapsed_time:.4f} seconds")

    if not X: # If the list X is empty (no images processed successfully)
        return np.array([]), [] # Return empty NumPy arrays

    return np.array(X), file_names

# Define the PDF creation function
def create_pdf_report(results, report_title="Image Processing Report"):
    """
    Generates a PDF report from the processing results.

    Args:
        results (list of dict): A list of dictionaries, where each dict contains
                                information for one image's prediction with BOTH
                                logistic and CNN results.
        report_title (str): The main title of the report.

    Returns:
        io.BytesIO: An in-memory binary stream containing the PDF content.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14, style="B") # Larger title
    
    pdf.cell(200, 10, txt=report_title, ln=True, align="C")
    pdf.ln(10) # Line break

    pdf.set_font("Arial", size=10)
    
    # Get current time in your specific time zone
    # Use the remembered timezone information
    current_location_timezone = pytz.timezone('America/Sao_Paulo') # Using remembered location's timezone
    current_time_str = datetime.now(current_location_timezone).strftime('%Y-%m-%d %H:%M:%S %Z%z')
    
    pdf.cell(200, 10, txt=f"Report Date: {current_time_str}", ln=True)
    pdf.ln(5)

    if not results: # This check handles an empty list correctly
        pdf.cell(200, 10, txt="No results available for this report.", ln=True)
    else:
        for result in results:
            pdf.set_font("Arial", style="B", size=11)
            pdf.cell(200, 10, txt=f"Image: {result.get('file_name', 'N/A')}", ln=True)
            pdf.set_font("Arial", size=10)
            
            # Logistic Regression Result
            pdf.cell(200, 7, txt=f"  Quality Classification: {result.get('logistic_class_name', 'N/A')}", ln=True)
            
            # CNN Luminescence Result (using the CORRECTED key name)
            pdf.cell(200, 7, txt=f"  Luminescence Probability: {result.get('cnn_probability_luminescence', 'N/A')}", ln=True)
            pdf.cell(200, 7, txt=f"  Luminescence Predicted Class: {result.get('cnn_predicted_class_name', 'N/A')} (Label: {result.get('cnn_predicted_class_index', 'N/A')})", ln=True)
            pdf.ln(3) # Small line break between results

    # Output the PDF to a BytesIO object
    pdf_output = pdf.output(dest='S')
    return io.BytesIO(pdf_output)