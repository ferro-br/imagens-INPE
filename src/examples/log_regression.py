# https://medium.com/@MudSnail/the-importance-of-logistic-regression-in-image-classification-1966d07e7a0c

# Step 1 - Import Libraries import numpy as np import matplotlib.pyplot as plt import seaborn as sns
import utils
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

utils.cls()

# Step 2 - Load the data
start_time = utils.tic()
print("Loading the MNIST 784 dataset, please wait... ", end="")
mnist = fetch_openml('mnist_784')
print("Done. ", end="")
utils.tac(start_time)

# Step 3 - Let's check the shape of the data
print("Size of data:", mnist.data.shape)
print("Size of labels:", mnist.target.shape)

# Turn the first image in MNIST data to a numpy object
some_digit = mnist.data.to_numpy()[0] 

# Reshape to plot the image (pixels 28x28)
some_digit_image = some_digit.reshape(28, 28)

# Plot our selected number with the label
plt.imshow(some_digit_image, cmap=plt.cm.binary, interpolation='nearest')
plt.axis()
plt.title(mnist.target[0])
plt.show()

# Split the MNIST data into training and test data
train_image, test_image, train_label, test_label = train_test_split(mnist.data, mnist.target, test_size=1/7, random_state=42)


# Instantiate model
# Here I changed the solver for it to handle multinomial loss
print("Creating the model... ", end="")
start_time = utils.tic()
log_regression = LogisticRegression(solver='saga')
print("Done. ", end="")
utils.tac(start_time)

# Fit the model and train
print("Fitting the model and training... ", end="")
start_time = utils.tic()
log_regression.fit(train_image, train_label)
print("Done. ", end="")
utils.tac(start_time)

#Let us measure performance
score = log_regression.score(test_image, test_label)
print("The accuracy score is:", score*100)