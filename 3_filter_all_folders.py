'''
THIS IS A CODE THAT FILTERS THE SEGMENTED CELLS BASED ON THE MASK DRAWN with 2_masking.py
SAVING FILTERED DATA IN A NEW FOLDER
HOW TO USE:

!!!!!!
root_folder: folder containing all data to segment stored in multiple folders
!!!!!!

'''

import os
from src.filtering import load_mask, load_centroids, save_filtered_centroids, filter_centroids_with_mask

# Define the root folder containing subfolders
root_folder = r"E:\VSC_SSD\MicroNeuSeg\data"


# Process all subfolders
channels = ["Iba1", "PV", "NeuN"]

for root, dirs, files in os.walk(root_folder):
    if root == root_folder:
        continue
    mask_folder = os.path.join(root, "masks")
    csv_folder = os.path.join(root, "output_csvs")
    filtered_csv_folder = os.path.join(root, "filtered_csvs")
    os.makedirs(filtered_csv_folder, exist_ok=True)

    # Process all masks in the current subfolder
    if os.path.exists(mask_folder):
        for mask_file in os.listdir(mask_folder):
            if mask_file.endswith("_mask.png"):
                identifier = mask_file.replace("_mask.png", "")
                mask_path = os.path.join(mask_folder, mask_file)

                print(f"Filtering centroids for {identifier} in folder: {root}")

                # Load the mask
                mask = load_mask(mask_path)

                # Process each channel
                for channel in channels:
                    csv_file = f"{identifier}_{channel}_coordinates.csv"
                    csv_path = os.path.join(csv_folder, csv_file)

                    centroids = load_centroids(csv_path)

                    if centroids.empty:
                        print(f"No centroids found for {identifier}_{channel}. Skipping...")
                        continue

                    # Filter centroids with the mask
                    filtered_centroids = filter_centroids_with_mask(mask, centroids)

                    # Save filtered centroids
                    filtered_save_path = os.path.join(filtered_csv_folder, f"{identifier}_{channel}_filtered_coordinates.csv")
                    save_filtered_centroids(filtered_centroids, filtered_save_path)
