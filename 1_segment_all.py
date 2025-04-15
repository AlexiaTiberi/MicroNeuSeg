'''
THIS IS THE SEGMENTATION SCRIPT TO PROCESS ALL DATA STORED IN SUBFOLDERS (AS DESCRIBED IN README)

HOW TO USE:

!!!!!!
parent_folder: folder containing all data to segment stored in multiple folders
!!!!!!

The names of the images need to contain either Iba1, PV, NeuN which are used by the code to decide which pipeline to use (DoG filtering vs Stardist)

The pipeline can be optimized for any use using the config.py which contains parameters to be tweaked with to improve cell segmentation

visualize=False, keep it false otherwise all images will pop up and slow everything down, if you are interested in seeing the segmentation use the .ipynb

'''
import os
from src.segment_single_folder import process_single_folder

# folder containing all data to segment stored in multiple folders
parent_folder = r"E:\VSC_SSD\MicroNeuSeg\data"

# 🔄 Process each subfolder
for root, dirs, files in os.walk(parent_folder):
    if root == parent_folder:
        continue
    process_single_folder(root, visualize= False) # when running the code in .py put false otherwise the visualizeation of the bounding boxes sucks