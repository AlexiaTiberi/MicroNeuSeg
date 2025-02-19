import os
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

# 📂 Define folder paths
data_folder = r"E:\LAB_TIBERI\IMMUNO_INVIVO\TAKE2\3PG_RL_M_WT"
mask_folder = os.path.join(data_folder, "masks")
filtered_csv_folder = os.path.join(data_folder, "filtered_csvs")

# 📥 Load the mask
def load_mask(identifier, mask_folder):
    mask_path = os.path.join(mask_folder, f"{identifier}_mask.png")
    mask = io.imread(mask_path, as_gray=True)
    mask = (mask > 0).astype(int)  # Convert to binary mask
    return mask

# 📥 Load filtered centroids
def load_filtered_centroids(identifier, channel, csv_folder):
    csv_path = os.path.join(csv_folder, f"{identifier}_{channel}_filtered_coordinates.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"No filtered CSV found for {identifier}_{channel}")
        return pd.DataFrame(columns=["x", "y"])

# 🔍 Check centroids against the mask
def check_centroids_in_mask(mask, centroids):
    results = []
    for _, row in centroids.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            inside = mask[y, x] == 1
            results.append((x, y, inside))
    return results

# 📊 Visualize centroids and mask
def visualize_mask_with_centroids(mask, results):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask, cmap='gray')
    
    for x, y, inside in results:
        color = 'green' if inside else 'red'
        ax.scatter(x, y, color=color, s=50, edgecolors='black', label='Inside' if inside else 'Outside')
    
    plt.title("Centroids in Mask (Green: Inside, Red: Outside)")
    plt.axis("off")
    plt.show()

# 🚀 Run the validation
identifier = "3PG_RL_M_WT_1"  # Change this to the specific identifier you want to check
channel = "Iba1"  # Choose from "Iba1", "PV", or "NeuN"

# Load the mask and filtered centroids
mask = load_mask(identifier, mask_folder)
filtered_centroids = load_filtered_centroids(identifier, channel, filtered_csv_folder)

# Check centroids inside the mask
results = check_centroids_in_mask(mask, filtered_centroids)

# Visualize the result
visualize_mask_with_centroids(mask, results)
