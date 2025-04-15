import os
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from skimage import io
import cv2
import numpy as np
import re
# Group images by shared identifier
def group_images_by_id(folder_path):
    grouped_files = {}
    for file in os.listdir(folder_path):
        if file.split("\\")[-1].lower().endswith(('.tif', '.tiff')):
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
            grouped_files[identifier][channel] = os.path.join(folder_path, file)
    return grouped_files

# Load RGB image
def load_rgb_image(file_group):
    def normalize(img):
        return cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    iba1_img = normalize(cv2.imread(file_group.get("Iba1", ''), cv2.IMREAD_GRAYSCALE))
    pv_img = normalize(cv2.imread(file_group.get("PV", ''), cv2.IMREAD_GRAYSCALE))
    neun_img = normalize(cv2.imread(file_group.get("NeuN", ''), cv2.IMREAD_GRAYSCALE))

    rgb_image = np.stack((pv_img, neun_img, iba1_img), axis=-1)
    return rgb_image

def draw_freehand_mask_opencv(image):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    plt.title("Draw a Mask: Left-click and drag to draw, release to finish")

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    def onselect(verts):
        verts = np.array(verts, dtype=np.int32)
        cv2.fillPoly(mask, [verts], 1)  # 🚀 Super-fast polygon filling
        ax.imshow(mask, alpha=0.3, cmap='Reds')
        fig.canvas.draw_idle()

    lasso = LassoSelector(ax, onselect)
    plt.show()
    return mask

# Save the mask as a PNG
def save_mask_as_png(mask, identifier, output_folder):
    mask_image = (mask * 255).astype(np.uint8)
    save_path = os.path.join(output_folder, f"{identifier}_mask.png")
    cv2.imwrite(save_path, mask_image)
    print(f"Mask saved to: {save_path}")