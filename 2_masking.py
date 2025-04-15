'''
THIS IS THE CODE TO MASK A REGION OF THE IMAGE YOU MIGHT WANT TO SEGMENT. PROCESS ONE FOLDER AT THE TIME!

HOW TO USE:

!!!!!!
data_folder: folder corresponding to a single "mouse" so only one folder at the time
!!!!!!


THe output will be a mask.png corresponding to each single image, so it will be saved as image_mask.png, be always consistent on how you name your data


'''
import os
from src.masking import group_images_by_id, load_rgb_image, draw_freehand_mask_opencv, save_mask_as_png

# Path to the image folder
data_folder = r"E:\VSC_SSD\MicroNeuSeg\data\animal1"
output_folder = os.path.join(data_folder, "masks")
os.makedirs(output_folder, exist_ok=True)


# 🚀 Process images
grouped_files = group_images_by_id(data_folder)

for identifier, file_group in grouped_files.items():
    if all(ch in file_group for ch in ["Iba1", "PV", "NeuN"]):
        print(f"Processing image set {identifier}...")
        rgb_image = load_rgb_image(file_group)


        mask = draw_freehand_mask_opencv(rgb_image)


        # Save the mask after confirmation
        save_mask_as_png(mask, identifier, output_folder)

    else:
        print(f"Missing channels for {identifier}. Skipping...")
