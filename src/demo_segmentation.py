import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================
# PATH HANDLING
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # project root

IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "retina.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.png")
VESSEL_MASK_PATH = os.path.join(OUTPUT_DIR, "vessel_mask.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# LOAD IMAGE
# =========================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Retina image not found at {IMAGE_PATH}")

img = cv2.resize(img, (512, 512))

# =========================================
# PREPROCESSING
# =========================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# =========================================
# VESSEL SEGMENTATION
# =========================================
vessel_mask = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15,
    3
)

kernel = np.ones((3, 3), np.uint8)
vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel)

# ✅ SAVE BINARY MASK
cv2.imwrite(VESSEL_MASK_PATH, vessel_mask)

# =========================================
# SIMPLE SEVERITY (DEMO)
# =========================================
vessel_pixels = np.sum(vessel_mask == 255)
total_pixels = vessel_mask.size
severity_index = (vessel_pixels / total_pixels) * 100

if severity_index < 30:
    severity_label = "Normal"
elif severity_index < 60:
    severity_label = "Moderate"
else:
    severity_label = "Severe"

print(f"Severity Index: {severity_index:.2f}%")
print(f"Severity Level: {severity_label}")

# =========================================
# VISUALIZATION
# =========================================
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.title("Original Retina")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Enhanced (CLAHE)")
plt.imshow(enhanced, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Vessels\n{severity_label} ({severity_index:.1f}%)")
plt.imshow(vessel_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig(RESULTS_PATH, dpi=300)
plt.show()

print("Results saved to:", RESULTS_PATH)
print("Vessel mask saved to:", VESSEL_MASK_PATH)
