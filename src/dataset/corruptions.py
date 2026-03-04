import os
import cv2
import numpy as np

def apply_corruptions(image_path: str, output_dir: str) -> dict:
    """
    Generates corrupted versions of the given image.
    Returns a dictionary of {corruption_name: new_image_path}.
    """
    try:
        import albumentations as A
    except ImportError:
        print("warn: albumentations not installed, returning clean image only")
        return {"clean": image_path}
        
    os.makedirs(output_dir, exist_ok=True)
        
    image = cv2.imread(image_path)
    if image is None:
        return {"clean": image_path}
        
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    corruptions = {
        "gaussian_noise": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        "blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        "rotation": A.SafeRotate(limit=15, p=1.0),
        "compression": A.ImageCompression(quality_lower=10, quality_upper=30, p=1.0),
        "low_resolution": A.Downscale(scale_min=0.25, scale_max=0.5, p=1.0)
    }

    results = {"clean": image_path}
    
    for c_name, transform in corruptions.items():
        augmented = transform(image=image)
        corrupted_image = augmented['image']
        
        c_path = os.path.join(output_dir, f"{name}_{c_name}{ext}")
        cv2.imwrite(c_path, corrupted_image)
        results[c_name] = c_path
        
    return results
