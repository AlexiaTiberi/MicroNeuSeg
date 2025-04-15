'''
 THIS CODE BRINGS IT ALL TOGETHER, IMPORTS MASKS AND FILTERED DATA AND PERFORMES THE OPERATION FOR EACH SINGLE IMAGE
HOW TO USE:

!!!!!!
parent_directory: folder containing all data already segmented stored in multiple folders (after 1_)

savepath: folder in which to save all the data from the analysis
!!!!!!

pixel_size: insert correct pixel size to get measures in micron

threshold: distance in pixel to calculate the association between cells

visualize=True saves images to see all detected cells

n_iterations: is the number of times to run the synthetic analysis, in a sort of bootstrapping way

'''

import os
import timeit
from src.analysis import process_folder
from src.analysis2 import process_folder2
# ---------------- BATCH PROCESSING ---------------- #
# Set the path to the parent directory containing all the folders
parent_directory = r"E:\VSC_SSD\MicroNeuSeg\data"
savepath = r"E:\VSC_SSD\MicroNeuSeg\results"

def original_sampling():
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            process_folder(folder_path, savepath, pixel_size=0.7575750, threshold=15, visualize=False, n_iterations=1)

    print("Batch processing complete.")

def optimized_sampling():       
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            process_folder2(folder_path, savepath, pixel_size=0.7575750, threshold=15, visualize=False, n_iterations=1)

    print("Batch processing complete.")

orig_time = timeit.timeit(original_sampling, number=1)
opt_time = timeit.timeit(optimized_sampling, number=1)

print(f"Avg original sampling time: {orig_time / 10:.4f} seconds")
print(f"Avg optimized sampling time: {opt_time / 10:.4f} seconds")