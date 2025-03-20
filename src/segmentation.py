
def process_iba1_somas(image, min_area=100, max_area=1000, low_sigma=1, high_sigma=5, solidity_thresh=0.8, eccentricity_thresh=0.99):
    """
    Optimized DoG-based segmentation to detect Iba1+ microglial somas and PV with shape filtering and bounding box visualization.

    Parameters:
    - image: Input grayscale image.
    - min_area: Minimum object area to keep.
    - max_area: Maximum object area to keep.
    - low_sigma: Standard deviation for the smaller Gaussian kernel.
    - high_sigma: Standard deviation for the larger Gaussian kernel.
    - solidity_thresh: Minimum solidity to filter out elongated structures.
    - eccentricity_thresh: Maximum eccentricity to filter out elongated structures.

    Returns:
    - coordinates: List of centroids (x, y) of detected somas.
    - bounding_boxes: List of bounding boxes (x, y, width, height).
    - labels: Labeled segmentation mask.
    """
    from skimage import filters, measure, morphology, exposure
    import numpy as np

    # Step 1: Apply Difference of Gaussians (DoG) filter
    dog_filtered = filters.difference_of_gaussians(image, low_sigma=low_sigma, high_sigma=high_sigma)

    # Step 2: Normalize and Threshold
    dog_rescaled = exposure.rescale_intensity(dog_filtered, in_range='image', out_range=(0, 255))
    dog_8bit = dog_rescaled.astype(np.uint8)
    threshold = filters.threshold_otsu(dog_8bit)
    binary_mask = (dog_8bit > threshold).astype(np.uint8)

    # Step 3: Morphological cleaning
    binary_mask = morphology.binary_opening(binary_mask, morphology.disk(2))
    binary_mask = binary_mask.astype(np.uint8)

    # Step 4: Connected components and filtering
    labels = measure.label(binary_mask)
    properties = measure.regionprops(labels)

    # Extract centroids and bounding boxes
    coordinates = []
    bounding_boxes = []

    for prop in properties:
        if min_area <= prop.area <= max_area and prop.solidity >= solidity_thresh and prop.eccentricity <= eccentricity_thresh:
            # Centroid for saving
            coordinates.append((int(prop.centroid[1]), int(prop.centroid[0])))
            # Bounding box for visualization
            x, y, w, h = prop.bbox[1], prop.bbox[0], prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0]
            bounding_boxes.append((x, y, w, h))

    print(f"Number of detected cells: {len(coordinates)}")
    return coordinates, bounding_boxes, labels




def process_iba1_somas_1(image, min_area=100, max_area=1000, low_sigma=1, high_sigma=5, 
                        solidity_thresh=0.8, eccentricity_thresh=0.99, min_distance=10):
    
    from skimage import filters, measure, morphology, exposure
    import numpy as np
    from scipy.spatial import cKDTree
    """
    Optimized DoG-based segmentation to detect Iba1+ microglial somas with shape filtering 
    and bounding box visualization, ensuring a minimum centroid distance by removing the 
    smaller bounding box when two are too close. Optimized for large-scale image processing.

    Parameters:
    - image: Input grayscale image.
    - min_area: Minimum object area to keep.
    - max_area: Maximum object area to keep.
    - low_sigma: Standard deviation for the smaller Gaussian kernel.
    - high_sigma: Standard deviation for the larger Gaussian kernel.
    - solidity_thresh: Minimum solidity to filter out elongated structures.
    - eccentricity_thresh: Maximum eccentricity to filter out elongated structures.
    - min_distance: Minimum allowed distance between centroids.

    Returns:
    - coordinates: List of centroids (x, y) of detected somas.
    - bounding_boxes: List of bounding boxes (x, y, width, height).
    - labels: Labeled segmentation mask.
    """
    
    # Step 1: Apply Difference of Gaussians (DoG) filter
    dog_filtered = filters.difference_of_gaussians(image, low_sigma=low_sigma, high_sigma=high_sigma)

    # Step 2: Normalize and Threshold
    dog_rescaled = exposure.rescale_intensity(dog_filtered, in_range='image', out_range=(0, 255))
    dog_8bit = dog_rescaled.astype(np.uint8)
    threshold = filters.threshold_otsu(dog_8bit)
    binary_mask = (dog_8bit > threshold).astype(np.uint8)

    # Step 3: Morphological cleaning
    binary_mask = morphology.binary_opening(binary_mask, morphology.disk(2))
    binary_mask = binary_mask.astype(np.uint8)

    # Step 4: Connected components and filtering
    labels = measure.label(binary_mask)
    properties = measure.regionprops(labels)

    # Extract centroids, bounding boxes, and areas
    centroids = []
    bounding_boxes = []
    areas = []

    for prop in properties:
        if min_area <= prop.area <= max_area and prop.solidity >= solidity_thresh and prop.eccentricity <= eccentricity_thresh:
            centroid = (int(prop.centroid[1]), int(prop.centroid[0]))  # (x, y) format
            bounding_box = (prop.bbox[1], prop.bbox[0], prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0])  # (x, y, w, h)
            
            centroids.append(centroid)
            bounding_boxes.append(bounding_box)
            areas.append(prop.area)

    # Step 5: Apply distance constraint and remove smaller bounding boxes
    if len(centroids) > 1:
        centroids = np.array(centroids)
        tree = cKDTree(centroids)  # Build KD-tree for fast neighbor lookups
        close_pairs = tree.query_ball_tree(tree, r=min_distance)  # Find neighbors within min_distance
        
        to_remove = set()
        for i, neighbors in enumerate(close_pairs):
            if i in to_remove:
                continue
            for j in neighbors:
                if j in to_remove or j == i:
                    continue
                # Remove the one with the smaller bounding box area
                if areas[i] < areas[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
        
        # Filter out centroids and bounding boxes that do not meet distance criteria
        filtered_centroids = [c for i, c in enumerate(centroids) if i not in to_remove]
        filtered_bounding_boxes = [b for i, b in enumerate(bounding_boxes) if i not in to_remove]
    else:
        filtered_centroids = centroids
        filtered_bounding_boxes = bounding_boxes

    print(f"Number of detected cells after filtering: {len(filtered_centroids)}")
    return filtered_centroids, filtered_bounding_boxes, labels



def process_pv_somas(image, min_area=100, max_area=1000, low_sigma=1, high_sigma=5, solidity_thresh=0.8, eccentricity_thresh=0.99):
    """
    Optimized DoG-based segmentation to detect Iba1+ microglial somas and PV with shape filtering and bounding box visualization.

    Parameters:
    - image: Input grayscale image.
    - min_area: Minimum object area to keep.
    - max_area: Maximum object area to keep.
    - low_sigma: Standard deviation for the smaller Gaussian kernel.
    - high_sigma: Standard deviation for the larger Gaussian kernel.
    - solidity_thresh: Minimum solidity to filter out elongated structures.
    - eccentricity_thresh: Maximum eccentricity to filter out elongated structures.

    Returns:
    - coordinates: List of centroids (x, y) of detected somas.
    - bounding_boxes: List of bounding boxes (x, y, width, height).
    - labels: Labeled segmentation mask.
    """
    from skimage import filters, measure, morphology, exposure
    import numpy as np

    # Step 1: Apply Difference of Gaussians (DoG) filter
    dog_filtered = filters.difference_of_gaussians(image, low_sigma=low_sigma, high_sigma=high_sigma)

    # Step 2: Normalize and Threshold
    dog_rescaled = exposure.rescale_intensity(dog_filtered, in_range='image', out_range=(0, 255))
    dog_8bit = dog_rescaled.astype(np.uint8)
    threshold = filters.threshold_otsu(dog_8bit)
    binary_mask = (dog_8bit > threshold).astype(np.uint8)

    # Step 3: Morphological cleaning
    binary_mask = morphology.binary_opening(binary_mask, morphology.disk(2))
    binary_mask = binary_mask.astype(np.uint8)

    # Step 4: Connected components and filtering
    labels = measure.label(binary_mask)
    properties = measure.regionprops(labels)

    # Extract centroids and bounding boxes
    coordinates = []
    bounding_boxes = []

    for prop in properties:
        if min_area <= prop.area <= max_area and prop.solidity >= solidity_thresh and prop.eccentricity <= eccentricity_thresh:
            # Centroid for saving
            coordinates.append((int(prop.centroid[1]), int(prop.centroid[0])))
            # Bounding box for visualization
            x, y, w, h = prop.bbox[1], prop.bbox[0], prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0]
            bounding_boxes.append((x, y, w, h))

    print(f"Number of detected cells: {len(coordinates)}")
    return coordinates, bounding_boxes, labels