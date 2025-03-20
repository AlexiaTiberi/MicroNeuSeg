# Preprocess the channel image
from skimage import io
from skimage import exposure
import numpy as np
from skimage.restoration import denoise_tv_chambolle

def preprocess_channel(image_path, enhance_contrast=True):
   
    # Load image
    img = io.imread(image_path)
    
    # Normalize image
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    
    # Optionally enhance contrast
    if enhance_contrast:
        img = exposure.equalize_adapthist(img)
    
    return img

def preprocess_channel_iba(image_path, enhance_contrast=True, tv_weight=0.1, gamma=0.9, tv_chambolle=False):
    # Load image
    img = io.imread(image_path)
    
    # Percentile-based normalization
    pmin, pmax = 3, 99.8
    mi, ma = np.percentile(img, pmin), np.percentile(img, pmax)
    img = (img - mi) / (ma - mi + 1e-8)
    img = np.clip(img, 0, 1)  # Ensure values remain in range

    # Apply TV denoising
    if tv_chambolle:
        img = denoise_tv_chambolle(img, weight=tv_weight)
    
    # Contrast enhancement
    if enhance_contrast:
        img = exposure.equalize_adapthist(img)

    # Apply gamma correction
    img = exposure.adjust_gamma(img, gamma)

    return img


def preprocess_channel_pv(image_path, enhance_contrast=True, tv_weight=0.1, gamma=0.9, tv_chambolle=False):
    # Load image
    img = io.imread(image_path)
    
    # Percentile-based normalization
    pmin, pmax = 3, 99.8
    mi, ma = np.percentile(img, pmin), np.percentile(img, pmax)
    img = (img - mi) / (ma - mi + 1e-8)
    img = np.clip(img, 0, 1)  # Ensure values remain in range

    # Apply TV denoising
    if tv_chambolle:
        img = denoise_tv_chambolle(img, weight=tv_weight)
    
    # Contrast enhancement
    if enhance_contrast:
        img = exposure.equalize_adapthist(img)

    # Apply gamma correction
    img = exposure.adjust_gamma(img, gamma)

    return img