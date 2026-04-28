import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_mask(image_path):
    # Use PIL to read the image to prevent automatic color conversions 
    # that some libraries apply by default.
    mask = np.array(Image.open(image_path))

    # Print the unique values to numerically verify your instances saved correctly
    unique_values = np.unique(mask)
    print(f"Mask shape: {mask.shape}")
    print(f"Unique pixel values (Instance IDs): {unique_values}")

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # --- View 1: Auto-scaled Grayscale ---
    # imshow automatically scales the colors so the max value becomes white
    plt.subplot(1, 2, 1)
    plt.title("Auto-scaled Grayscale")
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04, label='Pixel Value')

    # --- View 2: Distinct Colormap ---
    # 'nipy_spectral' or 'tab20' are great for instance masks because 
    # they assign vastly different colors to sequentially close integers (1, 2, 3)
    plt.subplot(1, 2, 2)
    plt.title("Instance Colormap (High Contrast)")
    
    # Optional: Mask out the background (0) so it stays black, making instances pop
    masked_data = np.ma.masked_where(mask == 0, mask)
    
    # Set background color to black
    ax = plt.gca()
    ax.set_facecolor('black')
    
    plt.imshow(masked_data, cmap='nipy_spectral', interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04, label='Instance ID')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with the path to your saved mask
    image_path = 'Outputs/batch_5/val/frame_675/Replicator_instance_raw.png' 
    visualize_mask(image_path)