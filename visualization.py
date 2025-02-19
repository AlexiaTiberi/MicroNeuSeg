# File: visualization.py
import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_dog_results(image, bounding_boxes, labels, title="Iba1 DoG Segmentation Results"):
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    

    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.pause(1)
    plt.close()

def visualize_filtered_bounding_boxes(image, labels, filtered_coordinates, min_area, max_area, title="Visualization"):
    """
    Visualize the segmentation results showing bounding boxes of filtered cells only.
    - Red rectangles: Bounding boxes around filtered cells.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from skimage.measure import regionprops

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    
    # Draw bounding boxes for filtered cells
    for prop in regionprops(labels):
        min_row, min_col, max_row, max_col = prop.bbox
        # Check if the cell passes the size filter
        if min_area <= prop.area <= max_area:
            rect = patches.Rectangle(
                (min_col, min_row),  # Bottom-left corner
                max_col - min_col,   # Width
                max_row - min_row,   # Height
                linewidth=1.0,
                edgecolor="red",
                facecolor="none",
                alpha=0.8
            )
            ax.add_patch(rect)
    
    ax.axis("off")
    plt.pause(1)
    plt.close()


