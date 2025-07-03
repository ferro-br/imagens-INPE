import cv2
#from src.utils import *
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
    verbose: bool = True
) -> tuple[np.ndarray, list[str], list[np.ndarray], list[np.ndarray]]: 
    """
    Processes a list of Streamlit UploadedFile objects, extracts features,
    and returns a NumPy array of features and a list of file names.
    """
    X               = [] # list to store the samples (features, currently the same thing as "reduced_images")
    file_names      = [] # list to store the file names corresponding to the samples
    reduced_images  = [] # list to store resized images for reporting (currently, the same thing as "X")
    original_images = [] # list to store the original images 

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
            uploaded_file_obj.seek(0) # Rewind the file pointer to the beginning 
            # 1. Read the file content in bytes
            bytes_data = uploaded_file_obj.read()
            # 2. Open the bytes as a PIL Image (Pillow)
            image_pil = Image.open(io.BytesIO(bytes_data))

            # Convert PIL Image to OpenCV format (NumPy array) for resizing
            # Ensure it's in BGR format if your model expects it, and for cv2.imencode
            img_cv2 = np.array(image_pil)
            if img_cv2.ndim == 2: # Grayscale
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR)
            elif img_cv2.shape[2] == 4: # RGBA
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGBA2BGR)
            else: # RGB or BGR (ensure it's BGR for OpenCV functions)
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR) # PIL is RGB, OpenCV is BGR
       
            # Resize image for models and PDF display
            resized_img_cv2 = cv2.resize(img_cv2, (sample_dims[1], sample_dims[0]), interpolation=interpol_method)

            # 3. Call the NEW extract_features_from_pil function
            features = resized_img_cv2 # Note: currently, "features" and "resized_img_cv2" are the same exact thing
 
            X.append(features)
            file_names.append(current_file_name)
            reduced_images.append(resized_img_cv2) # Store the resized image
            original_images.append(img_cv2)  # Store the original image

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
        return np.array([]), [], [], [] # Return empty NumPy arrays

    return np.array(X), file_names, reduced_images, original_images

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
            image_data = result.get('image_data') 

            if image_data is not None and image_data.size > 0:
                try:
                    image_width_mm = 80 # Define the width for the image in the PDF
                    target_dpi = 200    # Define the DPI for quality

                    # Get a properly resized copy of the image
                    # Note: "resample_image_for_pdf" is pixeling the image whatever the reason
                    #resized_image_data_for_pdf = resample_image_for_pdf(image_data, image_width_mm, target_dpi)
                    resized_image_data_for_pdf = image_data

                    # Encode the pre-resized image data to PNG
                    # Use resized_image_data_for_pdf directly, it's already uint8
                    is_success, buffer = cv2.imencode(".png", resized_image_data_for_pdf) 
                    
                    if is_success:
                        if pdf.get_y() + image_width_mm + 10 > pdf.h - 20:
                            pdf.add_page()
                            # Optional: Add file name again on new page if it's the start of an entry
                            pdf.set_font("Arial", style="B", size=11)
                            pdf.cell(200, 10, txt=f"Image: {result.get('file_name', 'N/A')} (Continued)", ln=True)
                            pdf.set_font("Arial", size=10)

                        pdf.image(io.BytesIO(buffer), x=pdf.get_x() + 5, y=pdf.get_y(), w=image_width_mm)
                        pdf.set_xy(pdf.get_x(), pdf.get_y() + image_width_mm + 5)
                    else:
                        pdf.cell(200, 7, txt="  (Error: Could not encode image for PDF)", ln=True)
                except Exception as e:
                    pdf.cell(200, 7, txt=f"  (Error embedding image: {e})", ln=True)
            else:
                pdf.cell(200, 7, txt="  (No image data available for PDF display)", ln=True)

            # Logistic Regression Result
            pdf.cell(200, 7, txt=f"  Quality Classification: {result.get('log_pred_class', 'N/A')}", ln=True)
            
            # CNN Luminescence Result (using the CORRECTED key name)
            pdf.cell(200, 7, txt=f"  Wave patterns detected with probability: {result.get('cnn_pred_prob_val', 'N/A')}", ln=True)

            if result.get('log_pred_value', 'N/A') == 1:
              pdf.cell(200, 7, txt=f"  Predicted Class: {result.get('cnn_pred_class', 'N/A')} (Label: {result.get('cnn_pred_value', 'N/A')})", ln=True)
            else:
              pdf.cell(200, 7, txt=f"  Predicted Class: {result.get('cnn_pred_class', 'N/A')} (Irrelevant since the image is unusable)", ln=True)
            
            pdf.ln(3) # Small line break between results

    # Output the PDF to a BytesIO object
    pdf_output = pdf.output(dest='S')
    return io.BytesIO(pdf_output)