import os

main_folder_path = r"E:\LAB_TIBERI\IMMUNO_INVIVO\ROOT\new"

for folder_name in os.listdir(main_folder_path):
    folder_path = os.path.join(main_folder_path, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)
        
        # Iterate through each file in the folder
        for file_name in files:
            old_file_path = os.path.join(folder_path, file_name)

            # Check if the file ends with "_PV.tif"
            if file_name.endswith("_temp_PV.tif"):
                temp_file_name = file_name.replace("_temp_PV.tif","_NeuN.tif")
                temp_file_path = os.path.join(folder_path, temp_file_name)
                os.rename(old_file_path, temp_file_path)