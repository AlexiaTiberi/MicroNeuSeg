import os
import pandas as pd
from PIL import Image
import numpy as np
def masking_csv_file (filepath, pixel_size_mm):    
# density calculation
    #import the mask for normalization of these numbers
    png_files = [f for f in os.listdir(filepath) if f.endswith('.png')]
    if len(png_files) != 1:
        raise ValueError(f"Expected exactly one PNG file in the folder, but found {len(png_files)}.")
    else:
        mask_path = os.path.join(filepath, png_files[0])

    # Open and process the mask (assuming mask is black and white)
    mask_pil = Image.open(mask_path).convert('L')  # Convert mask to grayscale (L)
    mask = np.array(mask_pil) > 128  # Threshold to create a binary mask (white = 1, black = 0)
    true_count = np.sum(mask)
    area = true_count*pixel_size_mm*pixel_size_mm

    csv_files = [f for f in os.listdir(filepath) if f.endswith('.csv') and "NeuN" in f]
    if len(csv_files) != 1:
        raise ValueError(f"Expected exactly one csv file in the folder, but found {len(csv_files)}.")
    else:
        csv_path = os.path.join(filepath, csv_files[0])
    # Convert the centroids to a DataFrame for easier manipulation
    centroids_df = pd.read_csv(csv_path, header=0)

    # Ensure the centroid coordinates are within the valid range of the mask dimensions
    # Flip coordinates for correct index (image uses y,x while DataFrame might use x,y)
    valid_centroids = centroids_df[
        (centroids_df['y'] >= 0) & (centroids_df['y'] < mask.shape[0]) &  # Check row index (y)
        (centroids_df['x'] >= 0) & (centroids_df['x'] < mask.shape[1])    # Check column index (x)
    ]

    # Filter the valid centroids inside the mask (white areas of the mask)
    valid_centroids = valid_centroids[
        mask[valid_centroids['y'].astype(int), valid_centroids['x'].astype(int)] == 1  # Notice the flip here
    ]

    # Save valid centroids to a CSV file
    valid_centroids.to_csv(os.path.join(filepath, 'NeuN_cells_filtered.csv'), index=False)

    return valid_centroids, area

import cv2
import matplotlib.pyplot as plt
# Function to plot coordinate distributions with the image as the background
def plot_coordinates_with_image(channel_data, image, label, color):
    # Convert the image to RGB if it's grayscale (necessary for displaying as an image underlay)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
    
    # Create a figure and display the image as the background
    plt.figure(figsize=(20, 20))
    plt.imshow(image_rgb, cmap='gray', alpha=0.5)  # Set alpha for transparency of the image
    plt.scatter(channel_data['x'], channel_data['y'], color=color, s=10, alpha=0.7, label=label)
    #plt.xlabel('X Coordinate')
    #plt.ylabel('Y Coordinate')
    plt.title(f'{label} Coordinate Distribution with Image Underlay')
    plt.legend()
    plt.show()