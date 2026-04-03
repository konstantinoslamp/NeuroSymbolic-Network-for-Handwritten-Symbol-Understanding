"""
Script to clean original handwritten images.
It applies thresholding, cropping, and resizing to 28x28.
WARNING: This overwrites the original files!
"""

import os
import cv2
import numpy as np
import glob

DATA_DIR = 'src/data/symbols'
FOLDERS = ['plus', 'minus', 'times', 'divide']

def smart_preprocess(image):
    """
    Cleans the image:
    1. Thresholds to remove gray background noise.
    2. Crops to the bounding box of the symbol.
    3. Resizes to fit in 20x20.
    4. Centers in 28x28.
    """
    # 1. Thresholding: Anything darker than 50 becomes pure black (0)
    _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_TOZERO)
    
    # 2. Find Bounding Box
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
        
    x, y, w, h = cv2.boundingRect(coords)
    crop = thresh[y:y+h, x:x+w]
    
    # 3. Resize to fit in 20x20 box
    max_dim = 20
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w <= 0 or new_h <= 0: return None
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 4. Center in 28x28 canvas
    final_img = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    final_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    return final_img

def clean_folder(folder_name):
    folder_path = os.path.join(DATA_DIR, folder_name)
    
    # Find original images (not augmented ones)
    all_files = glob.glob(os.path.join(folder_path, "*.*"))
    originals = [f for f in all_files if "aug_" not in os.path.basename(f)]
    
    print(f"Cleaning {len(originals)} original images in '{folder_name}'...")
    
    for file_path in originals:
        # Read image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None: continue
        
        # Invert if needed (white background -> black background)
        if np.mean(image[:5, :5]) > 128:
            image = cv2.bitwise_not(image)
            
        # Clean
        clean_img = smart_preprocess(image)
        
        if clean_img is not None:
            # Overwrite original file
            cv2.imwrite(file_path, clean_img)
            print(f"  Fixed: {os.path.basename(file_path)}")
        else:
            print(f"  Skipped (empty): {os.path.basename(file_path)}")

if __name__ == "__main__":
    print("Starting cleanup of original images...")
    for folder in FOLDERS:
        if os.path.exists(os.path.join(DATA_DIR, folder)):
            clean_folder(folder)
    print("\nDone! Your original images are now 28x28 and clean.")