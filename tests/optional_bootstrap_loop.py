import os
from analysis_boot import process_folder2
import numpy as np 
import pandas as pd

def loop1():
    # ---------------- BATCH PROCESSING ---------------- #
    # Set the path to the parent directory containing all the folders
    parent_directory = r"E:\LAB_TIBERI\IMMUNO_INVIVO\DEBUG"
    savepath = r"E:\LAB_TIBERI\IMMUNO_INVIVO\OUTPUT_PROXI_5"
    output_animals = [] #indexing by folder (i)

    # Iterate through each subfolder and process
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            identifier, pv_bootstrap, neu_bootstrap= process_folder2(folder_path, savepath, pixel_size=0.7575758, threshold=15, n_iterations=3)
            output_animals.append([identifier, pv_bootstrap, neu_bootstrap])
            print(identifier)
    # print("Batch processing complete.")
    all_identifiers=[]
    for i in range(len(output_animals)):
        all_identifiers += output_animals[i][0]

    is_wt = np.zeros(len(all_identifiers), dtype=bool)
    for idx,s1 in enumerate(all_identifiers):
        if 'WT' in s1:
            is_wt[idx] = True 
    
    # print("Batch processing complete.")
    all_pv_perc=[]
    for i in range(len(output_animals)):
        all_pv_perc.append(output_animals[i][1][:,0])
    all_pv_perc = np.concatenate(all_pv_perc)

    all_pv_nnd=[]
    for i in range(len(output_animals)):
        all_pv_nnd.append(output_animals[i][1][:,1])
    all_pv_nnd = np.concatenate(all_pv_nnd)


    # print("Batch processing complete.")
    all_neu_perc=[]
    for i in range(len(output_animals)):
        all_neu_perc.append(output_animals[i][2][:,0])
    all_neu_perc = np.concatenate(all_neu_perc)

    all_neu_nnd=[]
    for i in range(len(output_animals)):
        all_neu_nnd.append(output_animals[i][2][:,1])
    all_neu_nnd = np.concatenate(all_neu_nnd)    


    is_not_wt = np.logical_not(is_wt)

    pv_perc = all_pv_perc[is_wt,:].mean(axis=0) - all_pv_perc[is_not_wt, :].mean(axis=0)
    neu_perc = all_neu_perc[is_wt,:].mean(axis=0) - all_neu_perc[is_not_wt, :].mean(axis=0)
    
    pv_nnd = all_pv_nnd[is_wt,:].mean(axis=0) - all_pv_nnd[is_not_wt, :].mean(axis=0)
    neu_nnd = all_neu_nnd[is_wt,:].mean(axis=0) - all_neu_nnd[is_not_wt, :].mean(axis=0)


    np.savetxt(os.path.join(savepath, "output_pv_perc_1.csv"), pv_perc, delimiter=",", fmt='%g')
    np.savetxt(os.path.join(savepath, "output_neun_perc_1.csv"), neu_perc, delimiter=",", fmt='%g')

    np.savetxt(os.path.join(savepath, "output_pv_nnd_1.csv"), pv_nnd, delimiter=",", fmt='%g')
    np.savetxt(os.path.join(savepath, "output_neun_nnd_1.csv"), neu_nnd, delimiter=",", fmt='%g')

    return pv_perc, neu_perc, pv_nnd, neu_nnd


if __name__ == "__main__":
    loop1()