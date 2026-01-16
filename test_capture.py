import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a thin pattern frame and its inverse (e.g., col_9 and col_9_inv)
img_normal = cv2.imread("frames/frame_18.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img_inverse = cv2.imread("frames/frame_19.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Compute the difference
difference = img_normal - img_inverse

# Visualize: Everything > 0 is "White", everything < 0 is "Black"
# This removes the "washout" caused by metal bounces
recovered_pattern = np.zeros_like(difference)
recovered_pattern[difference > 0] = 255

plt.imshow(recovered_pattern, cmap='gray')
plt.title("Recovered Pattern (Differential)")
plt.show()