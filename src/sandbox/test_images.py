from utils import *
cls()

'''
# Example usage:
image_path = "teste.jpg"
new_size = (224, 224)  # Adjust to your desired size
resized_image = resize_image(image_path, new_size)

# Convert to NumPy array and flatten for logistic regression:
image_array = np.array(resized_image).flatten()


'''

# Load the image
# %%
import cv2 
img = cv2.imread('./img/cachorro.jpg')

print(get_files_by_ext('./_Amostras/clear1', '.tif'))
input("Press Enter to continue...")

# Resize the image to a specific size (width, height)
resized_img = cv2.resize(img, (50, 50))  # Adjust the size as needed

# Display the resized image
cv2.imshow('Image', img)
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
