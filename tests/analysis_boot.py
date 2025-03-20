import pandas as pd
import numpy as np
from scipy.spatial import distance
import os
import cv2
import matplotlib.pyplot as plt

def bootstrap_microglia_analysis_matching_avg_distance(coords_a, coords_b, mask, threshold, n_iterations=1000, max_iterations=1):

    def minimum_distances(matrix1, matrix2):
        """
        Calculate the minimum distance from each point in matrix1 to any point in matrix2.
    
        Parameters:
            matrix1 (np.ndarray): n1 x 2 matrix of coordinates.
            matrix2 (np.ndarray): n2 x 2 matrix of coordinates.
    
        Returns:
            np.ndarray: Array of minimum distances for each point in matrix1.
        """
        # Compute pairwise distances between points in matrix1 and matrix2
        distances = np.linalg.norm(matrix1[:, np.newaxis, :] - matrix2[np.newaxis, :, :], axis=2)
    
        # Find the minimum distance for each point in matrix1
        min_distances = np.min(distances, axis=1)
    
        return min_distances   
   
    def sample_with_adjusted_min_distance(valid_indices, n_samples, target_avg_distance, max_iterations):
        """
        Adjust min_distance iteratively to match target average distance.
        """
        min_distance = 0  # Start with no constraint
        valid_indices = np.array(valid_indices)
        
        for _ in range(max_iterations):
            selected_coords = []
            candidates = valid_indices.copy()
            np.random.shuffle(candidates)
            
            for candidate in candidates:
                if len(selected_coords) == 0:  # Always add the first point
                    selected_coords.append(candidate)
                else:
                    # Use vectorized computation to find distances to all selected points
                    distances = np.linalg.norm(np.array(selected_coords) - candidate, axis=1)
                    if np.all(distances >= min_distance):
                        selected_coords.append(candidate)
                if len(selected_coords) >= n_samples:
                    break
            
            selected_coords = np.array(selected_coords)
            
            # Calculate average nearest neighbor distance for bootstrapped coordinates
            if len(selected_coords) > 1:
                boot_distances = distance.cdist(selected_coords, selected_coords, metric="euclidean")
                np.fill_diagonal(boot_distances, np.inf)  # Ignore self-distances
                boot_avg_distance = np.mean(np.min(boot_distances, axis=1))
            else:
                boot_avg_distance = 0
            
            # Adjust min_distance based on the difference
            if np.isclose(boot_avg_distance, target_avg_distance, atol=5):  # Allow small tolerance
                return selected_coords
            min_distance += (target_avg_distance - boot_avg_distance) * 0.1  # Adjust proportionally
        
        return selected_coords

    # Ensure the mask is binary
    mask = mask > 0

    # Get valid pixel coordinates from the mask
    valid_indices = np.argwhere(mask)  # Shape: (N, 2) for (row, col)

    # Calculate average nearest neighbor distance for real microglia
    true_distances = distance.cdist(coords_a, coords_a, metric="euclidean")
    np.fill_diagonal(true_distances, np.inf)  # Ignore self-distances
    true_avg_distance = np.mean(np.min(true_distances, axis=1))
   
    proximity_counts = np.zeros(n_iterations)
    avg_nearest_distances = np.zeros(n_iterations)
    bootstrapped_coords_list = np.zeros([n_iterations,coords_a.shape[0], 2]) # SHApe n.iter, coordinates, number of cells
     
    for i in range(n_iterations):
        # Bootstrap sampling with adjusted min_distance
        bootstrap_coords_a = sample_with_adjusted_min_distance(valid_indices, len(coords_a), true_avg_distance, max_iterations)
        nearest_distance= minimum_distances(coords_b, bootstrap_coords_a)
        mask_threshold = nearest_distance<threshold
        
        proximity_counts[i]= mask_threshold.sum()
        avg_nearest_distances[i]=nearest_distance.mean()
        bootstrapped_coords_list[i,:,:]=bootstrap_coords_a

    # Calculate means
    return {
        "proximity_counts": proximity_counts,
        "avg_nearest_distances": avg_nearest_distances,
    }

def calculate_proximity_index_and_nearest_distance(channel_a, channel_b, threshold):
    """
    Calculates the proximity index and the average nearest distance between
    Channel B cells and Channel A cells.
    """
    coords_a = channel_a[['x', 'y']].values
    coords_b = channel_b[['x', 'y']].values
    
    proximity_counts = 0
    nearest_distances = []

    for cell_a in coords_a:
        distances = distance.cdist([cell_a], coords_b, metric='euclidean')[0]
        nearest_distance = np.min(distances)
        nearest_distances.append(nearest_distance)
        if nearest_distance <= threshold:
            proximity_counts += 1

    avg_nearest_distance = np.mean(nearest_distances) if nearest_distances else np.nan
    return proximity_counts, avg_nearest_distance

def load_csv(file_group):

    iba1_csv = pd.read_csv(file_group.get("Iba1", ''), header=0)
    pv_csv = pd.read_csv(file_group.get("PV", ''), header=0)
    neun_csv = pd.read_csv(file_group.get("NeuN", ''), header=0)


    return iba1_csv, pv_csv, neun_csv


def process_folder2(filepath, savepath, pixel_size=0.7575758, threshold=15,  n_iterations=1000):
    csv_path = os.path.join(filepath, "filtered_csvs")
    mask_path = os.path.join(filepath, "masks")
    pixel_size_mm = pixel_size / 1000

    # Step 1: Group CSV files by identifier and channel
    grouped_files = {}
    for file in os.listdir(csv_path):
        if file.lower().endswith('.csv'):
            parts = file.split("_")
            identifier = "_".join(parts[:-3])

            # Determine channel type
            if "iba1" in file.lower():
                channel = "Iba1"
            elif "pv" in file.lower():
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

            if "iba1" in file.lower():
                channel = "Iba1"
            elif "pv" in file.lower():
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
    file_counter=0
    all_pv_boot = np.zeros([len(grouped_files.items()), 2, n_iterations])#identifiers, 2:percent/averagenearestdistance, n_iterations
    all_neu_boot = np.zeros([len(grouped_files.items()),2, n_iterations])
    all_identifiers = []
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

             # Load mask and calculate area
            mask = cv2.imread(files["Mask"], cv2.IMREAD_GRAYSCALE)
            mask = mask > 128
            area = np.sum(mask) * pixel_size_mm * pixel_size_mm

            
            coords_microglia = iba1_data[['x', 'y']].values
            coords_pv = pv_data[['x', 'y']].values
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

            # Monte Carlo-like analysis for Microglia-associated PV and NeuN
 
            bootstrap_results_pv = bootstrap_microglia_analysis_matching_avg_distance(coords_microglia, coords_pv, mask, threshold, n_iterations=n_iterations)
            bootstrap_results_neun = bootstrap_microglia_analysis_matching_avg_distance(coords_microglia, coords_neu_filtered, mask, threshold, n_iterations=n_iterations)

            #percent_pv_bootstrap = (bootstrap_results_pv["proximity_mean"] / pv_data.shape[0])* 100
            percent_pv_bootstrap = (bootstrap_results_pv["proximity_counts"]/pv_data.shape[0])* 100
            percent_neu_bootstrap = (bootstrap_results_neun["proximity_counts"]/neun_data.shape[0])* 100
            nnd_pv=bootstrap_results_pv["avg_nearest_distances"]/pixel_size
            nnd_neun=bootstrap_results_neun["avg_nearest_distances"]/pixel_size
            
            all_pv_boot[file_counter, 0, :] = percent_pv_bootstrap
            all_pv_boot[file_counter, 1, :] = nnd_pv

            all_neu_boot[file_counter, 0, :]=percent_neu_bootstrap
            all_neu_boot[file_counter, 1, :]=nnd_neun


            all_identifiers.append(identifier)
            file_counter+=1
        else:
            print(f"Missing channels for {identifier}. Skipping...")

    return all_identifiers, all_pv_boot, all_neu_boot