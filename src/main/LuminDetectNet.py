# LuminDetectNet.py (Luminescence Detection Network)
# Builds a cnn to tell INPE images into two classes: 
#    (1) "positive" (with the luminiscence patterns) 
#    (2) "negative" (without the patterns)
# CAVEAT, COWBOY: Both "positive" and "negative" samples have to be "clear shots"!!
from utils import *
from matplotlib import pyplot as plt
#from keras import layers, models
from utils_net import build_CNN

# Parameteres of the data used in the learning
path_pos_files = '..\..\_Samples\clear\clear1' # Location of the positive files (can contain subfolders)
path_neg_files = '..\..\_Samples\clear\clear2' # Location of the negative files (can contain subfolders)

POSITIVE_LABEL = 1 # positive label
NEGATIVE_LABEL = 0 # negative label
CLASS_NAMES    = ['POSITIVE', 'NEGATIVE'] # classes names

# Parameters of the features
SAMPLE_DIMS     = (50, 50) # A tuple to hold the pictures' size (dimensions in pixels) after resizeing (to extract the features)
PERC_TEST_SET   = 0.2 # Percentage of the original dataset that will be used to test the model
INTERPOL_METHOD = cv2.INTER_LINEAR # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4

# Gets all the image files
list_pos_files = get_files_by_ext(path_pos_files, 'tif') # files with positive samples
list_neg_files = get_files_by_ext(path_neg_files, 'tif') # files with negative samples

# Assembles the whole dataset in a random fashion (file names and their labels)
whole_data_set = merge_randomly_with_labels(list_pos_files, list_neg_files, POSITIVE_LABEL, NEGATIVE_LABEL) 

# Parameters of the CNN
EPOCHS_NUMBER    = 20 # Number of epochs (10 was the original setting)
BATCH_SIZE       = 32 # Number of samples per gradient update
SHUFFLE_DATA     = True # Whether to shuffle the training data before each epoch

# Convolutional layers (cl_) parameters. Those layers perform convolution
CL_INPUT_SHAPE   = SAMPLE_DIMS+(3, ) # value for the input_shape parameter. (3,) is a tuple with the number of color channels (R, G, B)
CL_NUM_FILTERS_1 = 32         # The number of filters or kernels.Each filter detects a specific feature in the input images
CL_NUM_FILTERS_2 = 64         # The number of filters or kernels.Each filter detects a specific feature in the input images
CL_KERNEL_SIZE   = (3, 3)     # Kernel size, or the dimensions of the filter

# Pooling layer (pl_) parameters
PL_POOL_SIZE     =(2, 2)      # Tuple to hold the pool size. The dimensions of the rectangular window used to downsample the input feature maps.

# Dense layer (dl_) parameters
DL_NUM_NEURONS = 64 # Number of the neurons in the hidden dense layer

# Name of the file on disk who will store the CNN created 
CONV_NET_FILE = 'luminiscence_cnn_model.keras'


# Extract the features of the samples (each "sample" is a file name of a picture)
X, y, samp, samp1, samp0, _ = extract_features_bulk(whole_data_set, SAMPLE_DIMS, INTERPOL_METHOD, False, True)
print(f"{samp0} samples have label 0 and {samp1} samples have label 1.")

# Split data into training and testing (validation) sets
# NOTE: the samples are randomly scrambled already!
split_index  = int(len(X) * (1-PERC_TEST_SET))
train_images = X[:split_index]   # Training images
train_labels = y[:split_index]   # Training labels
valid_images  = X[split_index:]  # Validation images
valid_labels  = y[split_index:]  # Validation labels

# Normalize pixel values to be between 0 and 1
train_images, valid_images = train_images / 255.0, valid_images / 255.0

################################################################
print("Creating the convolutional base...", end='')
# NOTE: the parameter 'input_shape=(32, 32, 3))' tells that the CNN takes as inputs tensors 
#       of shape (image_height, image_width, color_channels). Height and width are given in 
#       pixels, 'colorChannels' is 3 for color images (RGB) or 1 for grayscale images.
#       'input_shape' has to be (32, 32, 1) for grayscale images of the same size.

cnn_params = [
    { # Block 1: Corresponds to the 1st Conv2D layer and its following MaxPooling2D
        'CL_NUM_FILTERS': CL_NUM_FILTERS_1,
        'CL_KERNEL_SIZE': CL_KERNEL_SIZE, # Common kernel size
        'ACTIVATION': 'relu',             # Activation for the Conv2D layer
        'CL_INPUT_SHAPE': CL_INPUT_SHAPE, # Input shape for your atmospheric images
        'PL_POOL_SIZE': PL_POOL_SIZE      # Pool size for MaxPooling2D after this Conv2D
    },
    { # Block 2: Corresponds to 2nd Conv2D layer and its following MaxPooling2D
        'CL_NUM_FILTERS': CL_NUM_FILTERS_2, 
        'CL_KERNEL_SIZE': CL_KERNEL_SIZE,   # Same kernel size
        'ACTIVATION': 'relu',               # Activation for the Conv2D layer
        'PL_POOL_SIZE': PL_POOL_SIZE        # Pool size for MaxPooling2D after this Conv2D
    },
    { # Block 3: Corresponds to the last Conv2D layer)
        'CL_NUM_FILTERS': CL_NUM_FILTERS_2, 
        'CL_KERNEL_SIZE': CL_KERNEL_SIZE,
        'ACTIVATION': 'relu'
        # No 'PL_POOL_SIZE' or 'MP_STRIDES' explicitly needed here
    }
]

conv_net = build_CNN(cnn_params, DL_NUM_NEURONS, 'relu', 1, 'sigmoid')

start_time = tic()
print("Compiling and trainning the model...", end='')
conv_net.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = conv_net.fit(train_images, train_labels, epochs=EPOCHS_NUMBER, batch_size=BATCH_SIZE, 
                    shuffle=SHUFFLE_DATA, validation_data=(valid_images, valid_labels))


# Save the entire model to a .keras file
# The model can be loaded back from the .keras file as below:
# CONV_NET = models.load_model(CONV_NET_FILE)
conv_net.save(CONV_NET_FILE)
print(f"Model saved successfully to {CONV_NET_FILE}")

tac(start_time)

start_time = tic()
print("Evaluating the model...", end='')
valid_loss, valid_acc = conv_net.evaluate(valid_images, valid_labels, verbose=2)
tac(start_time)

print(f'You\'ve got an accuracy of {np.round(valid_acc*100, 2)}% in the validation dataset.')

plt.plot(history.history['accuracy'], label='Accuracy (training set)')
plt.plot(history.history['val_accuracy'], label = 'Accuracy (validation set)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
                    