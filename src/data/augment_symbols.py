"""
Data Augmentation Script for Handwritten Symbols
Features:
- Smart Preprocessing (Thresholding + Centering) to fix gray boxes & small symbols
- Gentle Augmentation
"""

import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import glob

# Configuration
TARGET_COUNT = 500
DATA_DIR = 'src/data/symbols'
FOLDERS = ['plus', 'minus', 'times', 'divide']

# --- AUGMENTATION PIPELINE ---
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.OneOf([
        A.Morphological(scale=(1, 1), operation='erosion', p=0.5),
        A.Morphological(scale=(1, 1), operation='dilation', p=0.5),
    ], p=0.3),
])

def smart_preprocess(image):
    """
    Cleans the image:
    1. Thresholds to remove gray background noise.
    2. Crops to the bounding box of the symbol.
    3. Resizes to fit in 20x20.
    4. Centers in 28x28.
    """
    # 1. Thresholding: Anything darker than 50 becomes pure black (0)
    # This kills the "gray box" effect
    _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_TOZERO)
    
    # 2. Find Bounding Box (Crop whitespace)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None # Empty image
        
    x, y, w, h = cv2.boundingRect(coords)
    crop = thresh[y:y+h, x:x+w]
    
    # 3. Resize to fit in 20x20 box (preserving aspect ratio)
    # We use 20x20 so there is padding in the 28x28 final image
    max_dim = 20
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w <= 0 or new_h <= 0: return None
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 4. Center in 28x28 canvas
    final_img = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate offsets to center it
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    
    final_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    return final_img

def augment_folder(folder_name):
    folder_path = os.path.join(DATA_DIR, folder_name)
    
    # Get existing images (exclude previously augmented ones)
    all_files = glob.glob(os.path.join(folder_path, "*.*"))
    seed_images = [f for f in all_files if "aug_" not in os.path.basename(f)]
    
    if not seed_images:
        print(f"No seed images found in {folder_name}. Skipping.")
        return

    # Clean up OLD augmentations
    for f in all_files:
        if "aug_" in os.path.basename(f):
            os.remove(f)

    print(f"Processing '{folder_name}'...")

    generated = 0
    pbar = tqdm(total=TARGET_COUNT)
    
    while generated < TARGET_COUNT:
        seed_path = np.random.choice(seed_images)
        
        # Read image as grayscale
        image = cv2.imread(seed_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None: continue
        
        # Invert if needed (we want white symbol on black background)
        # Check the corners: if corners are white, it's likely black-on-white paper
        if np.mean(image[:5, :5]) > 128:
            image = cv2.bitwise_not(image)

        # --- APPLY SMART PREPROCESSING ---
        clean_img = smart_preprocess(image)
        if clean_img is None: continue

        # --- APPLY AUGMENTATION ---
        try:
            augmented = transform(image=clean_img)['image']
            
            save_name = f"aug_{generated}_{os.path.basename(seed_path)}"
            save_path = os.path.join(folder_path, save_name)
            cv2.imwrite(save_path, augmented)
            
            generated += 1
            pbar.update(1)
        except Exception as e:
            pass

    pbar.close()

if __name__ == "__main__":
    print("Starting SMART Data Augmentation...")
    
    for folder in FOLDERS:
        if os.path.exists(os.path.join(DATA_DIR, folder)):
            augment_folder(folder)
        else:
            print(f"Folder not found: {folder}")
            
    print("\nDone! Images are now clean, centered, and 28x28.")