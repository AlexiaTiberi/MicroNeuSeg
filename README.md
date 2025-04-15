# MicroNeuSeg

**MicroNeuSeg** is a Python-based pipeline designed to perform spatial analysis of microglia and neurons in mouse brain imaging data. The pipeline processes multiple image channels—Microglia (cx3cr1gfp or Iba1), PV, and NeuN—and outputs the spatial coordinates of detected cells, enables region-of-interest masking, and allows extensive data filtering and visualization.

---

## 📋 Data Setup and Folder Structure

The pipeline expects data organized as follows:

![image](https://github.com/user-attachments/assets/77e92764-9886-4c79-a6cf-b5826a86a103)

Each subfolder represents an animal and contains TIFF images named with the following format:

<animal_name>_<section_number>_<marker>.tif

Markers should explicitly be named: Iba1, PV, or NeuN, though the code is not case sensitive.

## 🚀 Features


The pipeline consists of sequentially numbered scripts to be executed in order:

1. **0_segment_single_folder.ipynb** *(Optional but recommended)*
   - Tests segmentation configuration on data from a single animal folder - use "animal1" folder path as defined in the Data Setup and Folder Structure chapter.
   - Useful for interactive parameter adjustments via `config.py`.
   - Start with this to broadly set up the segmentation

2. **1_segment_all_folders.ipynb / 1_segment_all.py**
   - Performs segmentation simultaneously across all animal folders - use "data" folder path as defined in the Data Setup and Folder Structure chapter.
   - Interactive visualization recommended via notebook version (`.ipynb`).

3. **2_masking.ipynb**
   - Allows manual masking to define a specific region of interest (ROIs).
   - This is mandatory as this pipeline was constructed to analyze a mouse cortex, so there is a need to cut away empty parts of the images and also subcortical stuff... see the data folder for some sample images

4. **3_filter_all_folders.ipynb** 
   - Uses the mask drawn in 2 to filter the data coordinates from the segmentation

5. **4_extract_data.ipynb** 
   - This is the code that does the "data analysis"
   - It will output an excel file per section analyzed of all mice
Parameters:






---

## ⚙️ Installation & Requirements
**My personal suggestions is to first install stardist and tensorflow using their installation guidelines, the rest you can easily add later when you run the script**

### Prerequisites

- Python ≥ 3.8
- Stardist and TensorFlow environment ([installation guide](https://github.com/stardist/stardist))

### Other Python Libraries

- `scikit-image`
- `opencv-python`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.
