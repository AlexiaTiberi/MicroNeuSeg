import os
import pandas as pd
from skimage import io

# 📂 Define the root folder containing subfolders
root_folder = r"E:\LAB_TIBERI\IMMUNO_INVIVO\ROOT\newdata"

# 📥 Load mask
def load_mask(mask_path):
    mask = io.imread(mask_path, as_gray=True)
    mask = (mask > 0).astype(int)  # Convert to binary mask
    return mask

# 📥 Load centroids for a specific channel
def load_centroids(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=["x", "y"])

# 📤 Save filtered centroids
def save_filtered_centroids(filtered_centroids, save_path):
    filtered_centroids.to_csv(save_path, index=False)
    print(f"Filtered centroids saved to: {save_path}")

# 🚀 Filter centroids using the mask
def filter_centroids_with_mask(mask, centroids):
    filtered = []
    for _, row in centroids.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x]:  # Check if the point is inside the mask
                filtered.append((x, y))
    return pd.DataFrame(filtered, columns=["x", "y"])

# 🔄 Process all subfolders
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
