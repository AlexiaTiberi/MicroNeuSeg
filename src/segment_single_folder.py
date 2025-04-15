import os
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import importlib
import config
from src.preprocessing import preprocess_channel_pv, preprocess_channel_iba, preprocess_channel
from src.export import save_coordinates_to_csv
from src.segmentation import process_iba1_somas_1, process_pv_somas
from src.visualization import visualize_dog_results, visualize_filtered_bounding_boxes
from src.filtercoordinates import extract_coordinates_with_size_filter

importlib.reload(config)
channel_params = config.channel_params

folder_path=r"E:\VSC_SSD\MicroNeuSeg\data"
# Load the pre-trained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

def process_single_folder(folder_path, visualize=False):
    cell_counts = {"Iba1": 0, "PV": 0, "NeuN": 0}
    output_folder = os.path.join(folder_path, "output_csvs")
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    image_file_paths = [os.path.join(folder_path, f) for f in image_files]

    for f in image_file_paths:
        basename = os.path.splitext(os.path.basename(f))[0]

        # Processing NeuN Channel
        if "neun" in basename.lower():
            channel = "NeuN"
            print(f"Processing {channel} in {basename} with StarDist...")
            img = preprocess_channel(f)
            n_tiles = model._guess_n_tiles(img)
            labels, _ = model.predict_instances(
                normalize(img, 1, 99.8, axis=(0, 1)), n_tiles=n_tiles,
                prob_thresh=channel_params[channel]["prob_thresh"],
                nms_thresh=channel_params[channel]["nms_thresh"]
            )
            min_area = channel_params[channel]["min_area"]
            max_area = channel_params[channel]["max_area"]
            coordinates = extract_coordinates_with_size_filter(labels, min_area, max_area)
            cell_counts[channel] += len(coordinates)
            if visualize:
                visualize_filtered_bounding_boxes(img, labels, coordinates, min_area, max_area, title=f"{channel} Filtered Results")

            csv_path = os.path.join(output_folder, f"{basename}_coordinates.csv")
            save_coordinates_to_csv(coordinates, csv_path)

        # Processing Iba1 Channel
        elif "iba1" in basename.lower():
            channel = "Iba1"
            print(f"Processing {channel} in {basename} with DoG filtering...")
            img = preprocess_channel_iba(f, enhance_contrast=True, tv_weight=0.1, gamma=0.9, tv_chambolle=channel_params[channel]["chambolle"])
            coordinates, bounding_boxes, labels = process_iba1_somas_1(
                img,
                min_area=channel_params[channel]["min_area"],
                max_area=channel_params[channel]["max_area"],
                low_sigma=channel_params[channel]["low_sigma"],
                high_sigma=channel_params[channel]["high_sigma"],
                solidity_thresh=channel_params[channel]["solidity_thresh"],
                eccentricity_thresh=channel_params[channel]["eccentricity_thresh"],
                min_distance=channel_params[channel]["min_distance"],
            )
            cell_counts[channel] += len(bounding_boxes)
            if visualize:
                visualize_dog_results(img, bounding_boxes, labels, title=f"{channel} DoG Segmentation Results")

            csv_path = os.path.join(output_folder, f"{basename}_coordinates.csv")
            save_coordinates_to_csv(coordinates, csv_path)

        # Processing PV Channel
        elif "pv" in basename.lower():
            channel = "PV"
            print(f"Processing {channel} in {basename} with DoG filtering...")
            img = preprocess_channel(f)
            coordinates, bounding_boxes, labels = process_pv_somas(
                img,
                min_area=channel_params[channel]["min_area"],
                max_area=channel_params[channel]["max_area"],
                low_sigma=channel_params[channel]["low_sigma"],
                high_sigma=channel_params[channel]["high_sigma"],
                solidity_thresh=channel_params[channel]["solidity_thresh"],
                eccentricity_thresh=channel_params[channel]["eccentricity_thresh"]
            )
            cell_counts[channel] += len(bounding_boxes)
            if visualize:
                visualize_dog_results(img, bounding_boxes, labels, title=f"{channel} DoG Segmentation Results")

            csv_path = os.path.join(output_folder, f"{basename}_coordinates.csv")
            save_coordinates_to_csv(coordinates, csv_path)

    print("\nCell counts for folder:")
    for channel, count in cell_counts.items():
        print(f"{channel}: {count} cells")

if __name__ == "__main__":
    process_single_folder(folder_path)
