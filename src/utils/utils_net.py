
from keras import layers, models

def build_CNN(cnn_params, DL_NUM_NEURONS, DL_ACTIVATION, OUT_NEURONS, OUT_ACTIVATION):
    """
    Builds a convolutional neural network (CNN) model.

    :param cnn_params: A tuple/list where each element is a dictionary
                       containing the parameters for a convolutional block.
                       Each dictionary should contain:
                       - 'CL_NUM_FILTERS' (int): Number of convolutional filters.
                       - 'CL_KERNEL_SIZE' (tuple): Dimensions of the convolution kernel (e.g., (3, 3)).
                       - 'ACTIVATION' (str): Activation function name (e.g., 'relu').
                       - 'PL_POOL_SIZE' (tuple): Pool size for MaxPooling2D (e.g., (2, 2)).
                                                (Expected for all blocks, but only applied if not the last block).
                       - 'CL_INPUT_SHAPE' (tuple): Input shape of the images (e.g., (64, 64, 3)).
                                                   (ONLY required for the first dictionary/block).
    :param DL_NUM_NEURONS (int): Number of neurons in the hidden Dense layer (after Flatten).
    :param DL_ACTIVATION (str): Activation function for the hidden Dense layer (e.g., 'relu').
    :param OUT_NEURONS (int): Number of neurons in the last layer (output Dense layer)
    :param OUT_ACTIVATION (str): Activation function for the output Dense layer (e.g., 'sigmoid' for binary, 'softmax' for multi-class).
    :returns: A Keras Sequential model (uncompiled).
    :rtype: tensorflow.keras.models.Sequential
    """
    # Initializes a sequential model, meaning the layers are arranged in a linear stack.
    model = models.Sequential() 

    # Add the very first layer (mandatory, of course), including the input_shape
    block_params = cnn_params[0]
    model.add(layers.Conv2D(block_params['CL_NUM_FILTERS'], 
                            block_params['CL_KERNEL_SIZE'], 
                            activation=block_params['ACTIVATION'], 
                            input_shape=block_params['CL_INPUT_SHAPE']))    
    model.add(layers.MaxPooling2D(block_params['PL_POOL_SIZE'])) # Downsampling: Max Pooling reduces the spatial dimensions


    #  Loop through remaining layers (from index 1 onwards)
    num_layers= len(cnn_params)
    for i in range(1, num_layers): # Start loop from the second element (index 1)
        block_params = cnn_params[i]
        # In essence, while convolutional layers learn and extract features from the input data, pooling 
        # layers reduce the dimensionality of the feature maps (that's why it is called downsampling).
        model.add(layers.Conv2D(block_params['CL_NUM_FILTERS'], 
                            block_params['CL_KERNEL_SIZE'], 
                            activation=block_params['ACTIVATION']))
        if (i<num_layers-1):
            model.add(layers.MaxPooling2D(block_params['PL_POOL_SIZE'])) # Downsampling: Max Pooling reduces the spatial dimensions

    # Complete the model by feeding the last output tensor from the convolutional base (of shape (4, 4, 64)) 
    # into one or more Dense layers to perform classification. Dense layers take vectors as input (which are 1D), 
    # while the current output is a 3D tensor. First, you will flatten (or unroll) the 3D output to 1D, then add one 
    # or more Dense layers on top. CIFAR has 10 output classes, so you use a final Dense layer with 10 outputs.
    print("Adding Dense layers on top...", end='')
    model.add(layers.Flatten()) #  flattens the input, converting it into a one-dimensional array.
    model.add(layers.Dense(DL_NUM_NEURONS, activation=DL_ACTIVATION)) # fully connected layer with 64 neurons which receives input from all neurons in the previous layer.
    model.add(layers.Dense(OUT_NEURONS, activation=OUT_ACTIVATION))
    return model