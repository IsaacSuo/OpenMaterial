import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def inspect_gt(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    print(f"--- Inspecting: {path} ---")
    
    # Read as-is
    img = cv2.imread(path, -1)
    
    if img is None:
        print("Error: Failed to read image (cv2.imread returned None). Check format.")
        return

    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Min: {img.min()}")
    print(f"Max: {img.max()}")
    print(f"Mean: {img.mean():.4f}")
    
    # Visualization for Depth (uint16)
    if img.dtype == np.uint16 or (img.ndim == 2 and img.dtype == np.float32):
        print("Type: Likely Depth Map")
        
        # Normalize to 0-255 for vis
        if img.max() > 0:
            norm = (img - img.min()) / (img.max() - img.min())
            vis = (norm * 255).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
            
            save_path = path + "_vis.png"
            cv2.imwrite(save_path, vis_color)
            print(f"Saved visualization to: {save_path}")
        else:
            print("Image is all zeros, cannot visualize.")

    # Visualization for Normal (RGB)
    elif img.ndim == 3:
        print("Type: Likely Normal Map (or RGB)")
        # Normal maps are usually already visible
        print("Top-left pixel (BGR):", img[0,0])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the image file")
    args = parser.parse_args()
    
    inspect_gt(args.path)
