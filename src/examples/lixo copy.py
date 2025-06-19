from utils import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models

# Parameteres of the data used in the learning
path_pos_files = '..\..\_Samples\clear\clear1'
path_neg_files = '..\..\_Samples\clear\clear2'

positive_label = 1 # positive label
negative_label = 0 # negative label
class_names    = ['positive', 'negative'] # classes names

# Parameters of the features
sample_dims     = (50, 50) # A tuple pictures' size (dimensions in pixel) after resizeing (to extract the features)
perc_test_set   = 0.2 # Percentage of the original dataset that will be used to test the model
interpol_method = cv2.INTER_LINEAR # Interpolation: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4

# Gets all the image files
list_pos_files = get_files_by_ext(path_pos_files, 'tif') # files with positive samples
list_neg_files = get_files_by_ext(path_neg_files, 'tif') # files with negative samples

# Assembles the whole dataset in a random fashion, with labels
whole_data_set = merge_randomly_with_labels(list_pos_files, list_neg_files, positive_label, negative_label) 

# Parameters of the CNN
epochs_number = 10 # Number of epochs (10 was the original setting)
# Convolutional layers (cl_) parameters. Those layers perform convolution
cl_input_dims_1  = sample_dims+(3, ) # value for the input_shape parameter
cl_num_filters_1 = 32         # The number of filters or kernels.Each filter detects a specific feature in the input images
cl_num_filters_2 = 64         # The number of filters or kernels.Each filter detects a specific feature in the input images
cl_kernel_size   = (3, 3)     # Kernel size, or the dimensions of the filter

# Pooling layer (pl_) parameters
pl_pool_size=(2, 2)         # pool size. The dimensions of the rectangular window used to downsample the input feature maps.

# Dense layer (dl_) parameters
dl_num_neurons = 64 # Number of the neurons in the hidden dense layer


# Extract the features of the samples
X, y, samp, samp1, samp0 = extract_features_bulk(whole_data_set, sample_dims, interpol_method, False, True)
print(f"{samp0} samples have label 0 and {samp1} samples have label 1.")

# Split data into training and testing sets
split_index  = int(len(X) * (1-perc_test_set))
train_images = X[:split_index] 
train_labels = y[:split_index] 
test_images  = X[split_index:] 
test_labels  = y[split_index:] 

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

################################################################
print("Creating the convolutional base...", end='')
# NOTE: the parameter 'input_shape=(32, 32, 3))' tells that the CNN takes as inputs tensors 
#       of shape (image_height, image_width, color_channels). Height and width are given in 
#       pixels, 'colorChannels' is 3 for color images (RGB) or 1 for grayscale images.
#       'input_shape' has to be (32, 32, 1) for grayscale images of the same size.

# Initializes a sequential model, meaning the layers are arranged in a linear stack.
model = models.Sequential() 

# In essence, while convolutional layers learn and extract features from the input data, pooling 
# layers reduce the dimensionality of the feature maps (that's why it is called downsampling).
model.add(layers.Conv2D(cl_num_filters_1, cl_kernel_size, activation='relu', input_shape=cl_input_dims_1))
model.add(layers.MaxPooling2D(pl_pool_size)) # Downsampling: Max Pooling reduces the spatial dimensions
model.add(layers.Conv2D(cl_num_filters_2, cl_kernel_size, activation='relu'))
model.add(layers.MaxPooling2D(pl_pool_size))
model.add(layers.Conv2D(cl_num_filters_2, cl_kernel_size, activation='relu'))

# Complete the model by feeding the last output tensor from the convolutional base (of shape (4, 4, 64)) 
# into one or more Dense layers to perform classification. Dense layers take vectors as input (which are 1D), 
# while the current output is a 3D tensor. First, you will flatten (or unroll) the 3D output to 1D, then add one 
# or more Dense layers on top. CIFAR has 10 output classes, so you use a final Dense layer with 10 outputs.
print("Adding Dense layers on top...", end='')
model.add(layers.Flatten()) #  flattens the input, converting it into a one-dimensional array.
model.add(layers.Dense(dl_num_neurons, activation='relu')) # fully connected layer with 64 neurons which receives input from all neurons in the previous layer.
model.add(layers.Dense(1, activation='sigmoid'))

start_time = tic()
print("Compiling and trainning the model...", end='')
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs_number, 
                    validation_data=(test_images, test_labels))
tac(start_time)

start_time = tic()
print("Evaluating the model...", end='')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
tac(start_time)
print(test_acc)
                    