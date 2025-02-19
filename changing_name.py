import os
from PIL import Image

def create_multitiff(main_folder_path):
    # Iterate through folders inside main_folder_path
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # List all files in the folder
            files = os.listdir(folder_path)
            
            # Dictionary to store groups of files by prefix
            file_groups = {}
            
            # Group files by the prefix before '_ch00', '_ch01', '_ch02'
            for file in files:
                if file.endswith("_Iba1.tif") or file.endswith("_NeuN.tif") or file.endswith("_PV.tif"):
                    # Extract the prefix (before '_ch00', '_ch01', '_ch02')
                    prefix = file.split('_')[0]
                    
                    if prefix not in file_groups:
                        file_groups[prefix] = {'Iba1': None, 'NeuN': None, 'PV': None}
                    
                    if "_Iba1.tif" in file:
                        file_groups[prefix]['Iba1'] = os.path.join(folder_path, file)
                    elif "_NeuN.tif" in file:
                        file_groups[prefix]['NeuN'] = os.path.join(folder_path, file)
                    elif "_PV.tif" in file:
                        file_groups[prefix]['PV'] = os.path.join(folder_path, file)

            # Process each group of files
            counter = 1
            for prefix, channels in file_groups.items():
                # Ensure all three channels are present
                if all(channels[key] is not None for key in ['Iba1', 'NeuN', 'PV']):
                    # Load each channel image
                    ch0 = Image.open(channels['Iba1']) # Iba1
                    ch1 = Image.open(channels['NeuN']) # PV
                    ch2 = Image.open(channels['PV']) # NeuN
                    
                    # Create a list of channels as separate pages in a TIFF
                    pages = [ch0, ch1, ch2]
                    # Define new filename with folder_name and a counter
                    new_filename_iba = f"{folder_name}_{counter}_Iba1.tif"
                    new_filename_pv = f"{folder_name}_{counter}_PV.tif"
                    new_filename_neun = f"{folder_name}_{counter}_NeuN.tif"

                    ch0.save(os.path.join(folder_path, new_filename_iba), save_all=True)
                    ch1.save(os.path.join(folder_path, new_filename_pv), save_all=True)
                    ch2.save(os.path.join(folder_path, new_filename_neun), save_all=True)

                    counter += 1
                    # Save as multi-page TIFF

                    print(f"TIFFs saved in {folder_name}")
                else:
                    print(f"Missing channels for prefix {prefix} in {folder_name}")

# Example usage:
main_folder_path = r"E:\LAB_TIBERI\IMMUNO_INVIVO\ROOT\new"
create_multitiff(main_folder_path)