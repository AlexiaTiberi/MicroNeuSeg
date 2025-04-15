import os
import pandas as pd
from skimage import io

# Load mask
def load_mask(mask_path):
    mask = io.imread(mask_path, as_gray=True)
    mask = (mask > 0).astype(int)  # Convert to binary mask
    return mask

# Load centroids for a specific channel
def load_centroids(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=["x", "y"])

#  Save filtered centroids
def save_filtered_centroids(filtered_centroids, save_path):
    filtered_centroids.to_csv(save_path, index=False)
    print(f"Filtered centroids saved to: {save_path}")

#  Filter centroids using the mask
def filter_centroids_with_mask(mask, centroids):
    filtered = []
    for _, row in centroids.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x]:  # Check if the point is inside the mask
                filtered.append((x, y))
    return pd.DataFrame(filtered, columns=["x", "y"])