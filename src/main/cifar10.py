# Modified from https://www.tensorflow.org/tutorials/images/cnn

import matplotlib.pyplot as plt

import tensorflow as tf
from keras import datasets, layers, models
from utils import * 

start_time = tic()
print("Importing dataset...", end='')
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
tac(start_time)


# Normalize pixel values to be between 0 and 1
start_time = tic()
print("Normalizing samples...", end='')
train_images, test_images = train_images / 255.0, test_images / 255.0
tac(start_time)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Creates the convolutional base
start_time = tic()
print("Creating the convolutional base...", end='')
# NOTE: the parameter 'input_shape=(32, 32, 3))' tells that the CNN takes as inputs tensors 
#       of shape (image_height, image_width, color_channels). Height and width are given in 
#       pixels, 'colorChannels' is 3 for color images (RGB) or 1 for grayscale images.
#       'input_shape' has to be (32, 32, 1) for grayscale images of the same size.


epochs_number = 10 # Number of epochs (10 was the original setting)

# Convolutional layers (cl_) parameters. Those layers perform convolution
cl_input_dims_1  =(32, 32, 3) # value for the input_shape parameter
cl_num_filters_1 = 32         # The number of filters or kernels.Each filter detects a specific feature in the input images
cl_num_filters_2 = 64         # The number of filters or kernels.Each filter detects a specific feature in the input images
cl_kernel_size =(3, 3)      # Kernel size, or the dimensions of the filter

# Pooling layer (pl_) parameters
pl_pool_size=(2, 2)         # pool size. The dimensions of the rectangular window used to downsample the input feature maps.

# Dense layer (dl_) parameters
dl_num_neurons = 64 # Number of the neurons in the hidden dense layer

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

model.add(layers.Dense(10)) 
tac(start_time)

start_time = tic()
print("Compiling and trainning the model...", end='')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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
                    