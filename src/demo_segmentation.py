import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "retina.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
VESSEL_MASK_PATH = os.path.join(OUTPUT_DIR, "vessel_mask.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Retina image not found")

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(2.0, (8, 8))
enhanced = clahe.apply(gray)

vessel_mask = cv2.adaptiveThreshold(
    enhanced, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15, 3
)

kernel = np.ones((3, 3), np.uint8)
vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel)

cv2.imwrite(VESSEL_MASK_PATH, vessel_mask)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(enhanced, cmap="gray"); plt.title("Enhanced"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(vessel_mask, cmap="gray"); plt.title("Vessels"); plt.axis("off")
plt.show()

print("Vessel mask saved at:", VESSEL_MASK_PATH)
