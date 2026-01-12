import cv2
import numpy as np
import os
from pathlib import Path
from shutil import move

# --- CONFIG ---
SOURCE_DIR = "crops/train/guava"   # your main guava folder
DEST_DIR = "crops/train/guava"     # same folder (subfolders created)

LIGHT = "light_green"
MEDIUM = "medium_green"
DARK = "dark_green"

# Create folders if not exist
for f in [LIGHT, MEDIUM, DARK]:
    Path(os.path.join(DEST_DIR, f)).mkdir(exist_ok=True)

def classify_color(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract V = brightness
    brightness = np.mean(hsv[:, :, 2])

    # Thresholds (you can adjust)
    if brightness > 180:     # bright leaf
        return LIGHT
    elif brightness > 110:   # mid tone
        return MEDIUM
    else:                    # dark leaf
        return DARK

def main():
    files = [f for f in os.listdir(SOURCE_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(files)} guava images.")

    for i, f in enumerate(files, 1):
        path = os.path.join(SOURCE_DIR, f)
        category = classify_color(path)

        if category is None:
            print(f"[!] Skipped: {f}")
            continue

        new_path = os.path.join(DEST_DIR, category, f)
        move(path, new_path)

        print(f"[{i}/{len(files)}] {f} → {category}")

    print("\nFinished sorting images by colour.")

if __name__ == "__main__":
    main()
