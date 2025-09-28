import cv2
import numpy as np

def test_reference_image():
    print("Testing reference.jpg...")
    
    # Try to load the image
    img = cv2.imread("reference.jpg")
    if img is None:
        print("ERROR: Could not load reference.jpg")
        print("Make sure the file exists and is a valid image format")
        return False
    
    print(f"✓ Successfully loaded reference.jpg")
    print(f"  - Size: {img.shape[1]}x{img.shape[0]} pixels")
    print(f"  - Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
    
    # Convert to grayscale and test features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Test ORB features
    orb = cv2.ORB_create(nfeatures=1000)
    kp, desc = orb.detectAndCompute(gray, None)
    print(f"  - ORB features found: {len(kp) if kp else 0}")
    
    # Test template sizes
    h, w = gray.shape
    scales = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]
    valid_scales = []
    for scale in scales:
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        if scaled_w > 10 and scaled_h > 10 and scaled_w < 800 and scaled_h < 600:
            valid_scales.append(scale)
    
    print(f"  - Valid template scales: {len(valid_scales)} out of {len(scales)}")
    
    # Save a preview
    preview = cv2.resize(img, (200, 150))
    cv2.imwrite("reference_preview.jpg", preview)
    print("  - Saved preview as reference_preview.jpg")
    
    # Show basic image statistics
    print(f"  - Brightness (mean): {gray.mean():.1f}")
    print(f"  - Contrast (std): {gray.std():.1f}")
    print(f"  - Min/Max values: {gray.min()}/{gray.max()}")
    
    return True

if __name__ == "__main__":
    test_reference_image()