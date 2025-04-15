import pandas as pd
import numpy as np
from scipy.spatial import distance
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage import exposure

def bootstrap_microglia_analysis_matching_avg_distance_2(
    identifier, coords_bootstrap, coords_b, threshold):
    """
    Perform proximity analysis using bootstrapped coordinates.

    Args:
    - identifier: Unique identifier for the analysis (used for saving results).
    - coords_bootstrap: Bootstrapped microglia coordinates.
    - coords_b: Coordinates of neurons (PV or NeuN).
    - threshold: Distance threshold for proximity.

    Returns:
    - Dictionary containing proximity count and average nearest distance.
    """
    proximity_count = 0
    nearest_distances = []

    for cell_b in coords_b:
        distances = distance.cdist([cell_b], coords_bootstrap, metric="euclidean")[0]
        nearest_distance = np.min(distances)
        nearest_distances.append(nearest_distance)
        if nearest_distance <= threshold:
            proximity_count += 1

    avg_nearest_distance = np.mean(nearest_distances) if nearest_distances else np.nan

    return {
        "proximity_count": proximity_count,
        "avg_nearest_distance": avg_nearest_distance
    }

def improved_heuristic_sampling(valid_indices, n_samples, target_avg_distance, max_iterations=10, atol=50):
    min_distance = target_avg_distance * 0.5
    valid_indices = np.array(valid_indices)
    best_coords = None
    best_diff = np.inf

    for _ in range(max_iterations):
        selected_coords = []
        candidates = valid_indices.copy()
        np.random.shuffle(candidates)

        for candidate in candidates:
            if len(selected_coords) == 0:
                selected_coords.append(candidate)
            else:
                distances = np.linalg.norm(np.array(selected_coords) - candidate, axis=1)
                if np.all(distances >= min_distance):
                    selected_coords.append(candidate)
            if len(selected_coords) >= n_samples:
                break

        selected_coords = np.array(selected_coords)

        if len(selected_coords) > 1:
            boot_distances = distance.cdist(selected_coords, selected_coords, metric="euclidean")
            np.fill_diagonal(boot_distances, np.inf)
            boot_avg_distance = np.mean(np.min(boot_distances, axis=1))
        else:
            boot_avg_distance = 0

        diff = boot_avg_distance - target_avg_distance

        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_coords = selected_coords

        if abs(diff) <= atol:
            break

        adjustment_rate = 0.1
        if diff < 0:
            min_distance += adjustment_rate * abs(diff)
        else:
            min_distance -= adjustment_rate * abs(diff)

        min_distance = max(min_distance, 0.1)

    return best_coords

def load_csv(file_group):

    iba1_csv = pd.read_csv(file_group.get("Iba1", ''), header=0)
    pv_csv = pd.read_csv(file_group.get("PV", ''), header=0)
    neun_csv = pd.read_csv(file_group.get("NeuN", ''), header=0)


    return iba1_csv, pv_csv, neun_csv


def calculate_proximity_index_and_nearest_distance_cKD(channel_a, channel_b, threshold):
    from scipy.spatial import cKDTree
    import numpy as np
    """
    Calculates the proximity index and the average nearest distance between
    Channel B cells and Channel A cells using a KDTree for fast nearest neighbor search.
    """
    coords_a = channel_a[['x', 'y']].values
    coords_b = channel_b[['x', 'y']].values
    
    # Build a KDTree for the B channel coordinates
    tree = cKDTree(coords_b)
    
    # Query the tree for the nearest neighbor for each point in A
    nearest_distances, _ = tree.query(coords_a, k=1)
    
    # Count how many points in A have a nearest neighbor in B within the threshold
    proximity_counts = np.sum(nearest_distances <= threshold)
    
    # Compute the average nearest neighbor distance
    avg_nearest_distance = np.mean(nearest_distances) if nearest_distances.size > 0 else np.nan
    
    return proximity_counts, avg_nearest_distance


def process_folder2(filepath, savepath, pixel_size=0.7575758, threshold=15, visualize=False, n_iterations=1):
    import re
    csv_path = os.path.join(filepath, "filtered_csvs")
    mask_path = os.path.join(filepath, "masks")
    pixel_size_mm = pixel_size / 1000

    # Step 1: Group CSV files by identifier and channel
    grouped_files = {}
    for file in os.listdir(csv_path):
        if file.lower().endswith('.csv'):

            parts = file.split("_")
            identifier = "_".join(parts[:-3])

            match = re.search(r'_(PV)(?=_filtered_coordinates|\.\w+$)', file)     

            # Determine channel type
            if "iba1" in file.lower():
                channel = "Iba1"
            elif match:
                channel = "PV"
            elif "neun" in file.lower():
                channel = "NeuN"
            else:
                continue

            if identifier not in grouped_files:
                grouped_files[identifier] = {}
            grouped_files[identifier][f"{channel}_CSV"] = os.path.join(csv_path, file)

    # Step 2: Group images by identifier and channel
    for file in os.listdir(filepath):
        if file.lower().endswith((".tiff", '.tif')):
            parts = file.split("_")
            identifier = "_".join(parts[:-1])

            match = re.search(r'_(PV)(?=_filtered_coordinates|\.\w+$)', file)  

            if "iba1" in file.lower():
                channel = "Iba1"
            elif match:
                channel = "PV"
            elif "neun" in file.lower():
                channel = "NeuN"
            else:
                continue

            if identifier not in grouped_files:
                grouped_files[identifier] = {}
            grouped_files[identifier][f"{channel}_Image"] = os.path.join(filepath, file)

    # Step 3: Link masks to identifiers
    for identifier in grouped_files:
        mask_file = os.path.join(mask_path, f"{identifier}_mask.png")
        if os.path.exists(mask_file):
            grouped_files[identifier]['Mask'] = mask_file
        else:
            print(f"Mask not found for identifier {identifier}")

    # Step 4: Load and analyze data
    for identifier, files in grouped_files.items():
        required_keys = ["Iba1_CSV", "PV_CSV", "NeuN_CSV", "Iba1_Image", "PV_Image", "NeuN_Image", "Mask"]
        if all(key in files for key in required_keys):
            # Load CSVs
            iba1_data = pd.read_csv(files["Iba1_CSV"], header=0)
            pv_data = pd.read_csv(files["PV_CSV"], header=0)
            neun_data = pd.read_csv(files["NeuN_CSV"], header=0)
            channel_neu_img = cv2.imread(os.path.join(filepath, files["NeuN_Image"]), cv2.IMREAD_GRAYSCALE)
            channel_iba_img = cv2.imread(os.path.join(filepath, files["Iba1_Image"]), cv2.IMREAD_GRAYSCALE)
            channel_pv_img = cv2.imread(os.path.join(filepath, files["PV_Image"]), cv2.IMREAD_GRAYSCALE) 
            channel_neu_img = exposure.equalize_adapthist(channel_neu_img)
            channel_iba_img = exposure.equalize_adapthist(channel_iba_img)
            channel_pv_img = exposure.equalize_adapthist(channel_pv_img)

            # Load mask and calculate area
            mask = cv2.imread(files["Mask"], cv2.IMREAD_GRAYSCALE)
            mask = mask > 128
            area = np.sum(mask) * pixel_size_mm * pixel_size_mm
                    # Finding NeuN+ - PV- cells
            coords_microglia = iba1_data[['x', 'y']].values
            coords_pv = pv_data[['x', 'y']].values
            def plot_coordinates_with_image(channel_data, image, label, color):
                # Convert the image to RGB if it's grayscale (necessary for displaying as an image underlay)
               
                # Create a figure and display the image as the background
                plt.figure(figsize=(20, 20))
                plt.imshow(image, cmap='gray', alpha=0.5)  # Set alpha for transparency of the image
                plt.scatter(channel_data['x'], channel_data['y'], color=color, s=10, alpha=0.7, label=label)
                #plt.xlabel('X Coordinate')
                #plt.ylabel('Y Coordinate')
                plt.title(f'{label} Coordinate Distribution with Image Underlay')
                plt.savefig(os.path.join (savepath, f"{label}_{identifier}.png"))
                plt.close()
            # Define a spatial tolerance (in micrometers) between PV and NEUN centroids
            tolerance_distance = 10  # e.g., 5 micrometers
            
            # Function to check if NeuN+ cells are within the tolerance distance from any PV cell
            def is_neuN_pv_negative(neu_cell, coords_pv, tolerance_distance):
                # Calculate the distance from the NeuN+ cell to each PV cell
                distances = distance.cdist([neu_cell], coords_pv, metric='euclidean')[0]
                # If the minimum distance is greater than the tolerance, consider it as PV-negative
                return np.min(distances) > tolerance_distance
            
            # Apply the filter with tolerance for NeuN+ PV-negative cells
            filtered_neu_data_with_tolerance = neun_data[
                neun_data.apply(lambda row: is_neuN_pv_negative((row['x'], row['y']), coords_pv, tolerance_distance), axis=1)]
            coords_neu_filtered = filtered_neu_data_with_tolerance[['x', 'y']].values

            # Proximity analysis PVto micro or NeuN to micro
            proximity_count_pv, avg_nearest_distance_pv = calculate_proximity_index_and_nearest_distance_cKD(pv_data, iba1_data, threshold)
            proximity_count_neu, avg_nearest_distance_neu = calculate_proximity_index_and_nearest_distance_cKD(filtered_neu_data_with_tolerance, iba1_data, threshold)
            percent_pv_associated = (proximity_count_pv / pv_data.shape[0]) * 100 if pv_data.shape[0] > 0 else 0
            percent_neu_associated = (proximity_count_neu / filtered_neu_data_with_tolerance.shape[0]) * 100 if filtered_neu_data_with_tolerance.shape[0] > 0 else 0

            # MicrogliatoPV and MicrogliatoNeuN association
            microglia_pv_associated_count, PVMICRO_average_nearest_distance = calculate_proximity_index_and_nearest_distance_cKD(iba1_data, pv_data, threshold)
            microglia_neu_associated_count, neunMICRO_average_nearest_distance = calculate_proximity_index_and_nearest_distance_cKD(iba1_data, filtered_neu_data_with_tolerance, threshold)
            # Association percentages
            percent_microglia_pv_associated = (microglia_pv_associated_count / iba1_data.shape[0]) * 100 if iba1_data.shape[0] > 0 else 0
            percent_microglia_neu_associated = (microglia_neu_associated_count / iba1_data.shape[0]) * 100 if iba1_data.shape[0] > 0 else 0
            valid_indices = np.argwhere(mask)
            true_distances = distance.cdist(coords_microglia, coords_microglia, metric="euclidean")
            np.fill_diagonal(true_distances, np.inf)  # Ignore self-distances
            true_avg_distance = np.mean(np.min(true_distances, axis=1))
            # Monte Carlo-like analysis for Microglia-associated PV and NeuN
            bootstrap_coords = improved_heuristic_sampling(valid_indices, len(coords_microglia), true_avg_distance)
            syn_distances = distance.cdist(bootstrap_coords, bootstrap_coords, metric="euclidean")
            np.fill_diagonal(syn_distances, np.inf)  # Ignore self-distances
            syn_avg_distance = np.mean(np.min(syn_distances, axis=1))
            #print(f"the real av distance{true_avg_distance} and the synth is {syn_avg_distance}")
            bootstrap_results_pv = bootstrap_microglia_analysis_matching_avg_distance_2(identifier, bootstrap_coords, coords_pv, threshold)
            bootstrap_results_neun = bootstrap_microglia_analysis_matching_avg_distance_2(identifier, bootstrap_coords, coords_neu_filtered, threshold)
            
            percent_pv_bootstrap = (bootstrap_results_pv["proximity_count"] / pv_data.shape[0]) * 100 if pv_data.shape[0] > 0 else 0
            percent_neu_bootstrap = (bootstrap_results_neun["proximity_count"] / filtered_neu_data_with_tolerance.shape[0]) * 100 if filtered_neu_data_with_tolerance.shape[0] > 0 else 0
            # Montecalro-like analysis for PV- or NeuN associated microglia
            #bootstrap_results_PVMICRO = bootstrap_microglia_analysis_matching_avg_distance_2(identifier, channel_pv_img, savepath, coords_pv, coords_microglia,  mask, threshold, n_iterations=n_iterations)
            #bootstrap_results_neunMICRO = bootstrap_microglia_analysis_matching_avg_distance_2(identifier,channel_neu_img, savepath, coords_neu_filtered, coords_microglia, mask, threshold, n_iterations=n_iterations)

            #percent_pv_bootstrap_PVMICRO = (bootstrap_results_PVMICRO["proximity_mean"] / iba1_data.shape[0]) * 100 if iba1_data.shape[0] > 0 else 0
            #percent_neu_bootstrap_neunMICRO = (bootstrap_results_neunMICRO["proximity_mean"] / iba1_data.shape[0]) * 100 if iba1_data.shape[0] > 0 else 0

            if visualize:
                plot_coordinates_with_image(iba1_data, channel_iba_img, 'Iba1+ Cells', 'blue')
                plot_coordinates_with_image(pv_data, channel_pv_img, 'PV+ Cells', 'red')
                plot_coordinates_with_image(neun_data, channel_neu_img, 'NeuN+ Cells', 'green')               
            # Save results
            count = pd.DataFrame({
                "Area": [area],

                "Microglia_Density": [iba1_data.shape[0] / area],
            
                "PV_Density": [pv_data.shape[0] / area],
            
                "NeuN+PV-_Density": [filtered_neu_data_with_tolerance.shape[0] / area],

                "NeuN_Density": [neun_data.shape[0] / area],
    
                "Perc_Microglia_associated_PV_Observed": [percent_pv_associated],
                "Perc_PV_associated_Microglia_Synthetic": [percent_pv_bootstrap],  

                "Average_Nearest_Distance_PV_to_microglia_Observed": [avg_nearest_distance_pv / pixel_size],
                "Average_Nearest_Distance_PV_to_microglia_Synthetic": [bootstrap_results_pv["avg_nearest_distance"]  / pixel_size],
                
                "Perc_Microglia_associated_NeuNPV_Observed": [percent_neu_associated],
                "Perc_Microglia_associated_NeuNPV_Synthetic": [percent_neu_bootstrap], 

                "Average_Nearest_Distance_NeuNPV_to_microglia_Observed": [avg_nearest_distance_neu / pixel_size],
                "Average_Nearest_Distance_NeuNPV_to_microglia_Syntetic": [bootstrap_results_neun["avg_nearest_distance"]  / pixel_size],

                "Perc_PV_associated_Microglia": [percent_microglia_pv_associated],
                #"Proximity_MEAN_PVMICRO_BOOTSTRAP": [percent_pv_bootstrap_PVMICRO],

                "Average_Nearest_Distance_microglia_to_PV": [PVMICRO_average_nearest_distance  / pixel_size],
                #"Average_Nearest_Distance_PVMICRO_BOOT": [bootstrap_results_PVMICRO["distance_mean"]  / pixel_size],

                "Perc_NeuNPV_associated_Microglia": [percent_microglia_neu_associated],
                #"Proximity_MEAN_NEUNMICRO_BOOTSTRAP": [percent_neu_bootstrap_neunMICRO], 

                "Average_Nearest_Distance_microglia_to_NeuNPV": [neunMICRO_average_nearest_distance  / pixel_size],
                #"Average_Nearest_Distance_NeuNMICRO_BOOT": [bootstrap_results_neunMICRO["distance_mean"]  / pixel_size],

                "E_I_ratio": [(percent_microglia_neu_associated -percent_microglia_pv_associated)/(percent_microglia_neu_associated +percent_microglia_pv_associated) if (percent_microglia_neu_associated + percent_microglia_pv_associated) != 0 else np.nan],

                })

            output_filename = f"{identifier}.xlsx"
            output_path = os.path.join(savepath, output_filename)
            count.to_excel(output_path, index=False)
            print(f"Results saved for {identifier} at {output_path}")
        else:
            print(f"Missing channels for {identifier}. Skipping...")


