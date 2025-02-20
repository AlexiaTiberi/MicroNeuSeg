import os
from analysis import process_folder
# ---------------- BATCH PROCESSING ---------------- #
# Set the path to the parent directory containing all the folders
parent_directory = r"E:\LAB_TIBERI\IMMUNO_INVIVO\ROOT\newdata"
savepath = r"E:\LAB_TIBERI\IMMUNO_INVIVO\new_data_filter"

# Iterate through each subfolder and process
for folder_name in os.listdir(parent_directory):
    folder_path = os.path.join(parent_directory, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        process_folder(folder_path, savepath, pixel_size=0.7575750, threshold=15, visualize=False, n_iterations=1)

print("Batch processing complete.")
