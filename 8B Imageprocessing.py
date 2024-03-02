import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread(r"C:/Users/kamini/Downloads/images/puppy.jpg")

# Convert BGR to RGB for display
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define kernel for morphological operations
kernel = np.ones((2, 2), np.uint8)

# Apply morphological closing
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Apply dilation
sure_bg = cv2.dilate(closing, kernel, iterations=3)

# Display the images
plt.subplot(211), plt.imshow(closing, 'gray')
plt.title("morphologyEx: Closing: 2x2"), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(sure_bg, 'gray')
plt.imsave(r'dilation.png', sure_bg)
plt.title("Dilation"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
