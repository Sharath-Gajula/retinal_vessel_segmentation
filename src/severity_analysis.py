import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MASK_PATH = os.path.join(PROJECT_ROOT, "output", "vessel_mask.png")

vessel_mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
if vessel_mask is None:
    raise FileNotFoundError("Run demo_segmentation.py first")

vessel_pixels = np.count_nonzero(vessel_mask)
total_pixels = vessel_mask.size
density = (vessel_pixels / total_pixels) * 100

dist = cv2.distanceTransform(vessel_mask, cv2.DIST_L2, 5)
thickness = 2 * np.mean(dist[vessel_mask > 0])

severity_index = (0.7 * density) + (0.3 * min(thickness * 10, 100))

if severity_index < 30:
    label = "Normal"
elif severity_index < 60:
    label = "Moderate"
else:
    label = "Severe"

print(f"Vessel Density: {density:.2f}%")
print(f"Avg Thickness: {thickness:.2f}")
print(f"Severity Index: {severity_index:.2f}")
print(f"Severity Level: {label}")



# import cv2
# import numpy as np
# import os

# # 👉 Segmentation = WHAT vessels look like
# # 👉 Severity Analysis = HOW serious the condition is

# # 👉 Take a retinal image
# # 👉 Extract blood vessels
# # 👉 Save vessel mask
# # 👉 Show result visually
# # 👉 Take binary vessel mask
# # 👉 Extract medical features
# # 👉 Calculate Severity Index

# # Severity Index = 18 → Normal
# # Severity Index = 45 → Moderate
# # Severity Index = 78 → Severe

# # =========================================
# # PATH HANDLING
# # =========================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(BASE_DIR)

# VESSEL_MASK_PATH = os.path.join(PROJECT_ROOT, "output", "vessel_mask.png")


# # =========================================
# # METRICS
# # =========================================
# def calculate_vessel_density(vessel_mask):
#     vessel_pixels = np.count_nonzero(vessel_mask)
#     total_pixels = vessel_mask.size
#     return (vessel_pixels / total_pixels) * 100


# def calculate_average_thickness(vessel_mask):
#     binary = np.where(vessel_mask > 0, 255, 0).astype(np.uint8)
#     dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

#     thickness = 2 * np.mean(dist_transform[binary == 255])
#     return thickness


# def calculate_severity_index(vessel_mask):
#     density = calculate_vessel_density(vessel_mask)
#     thickness = calculate_average_thickness(vessel_mask)

#     thickness_score = min((thickness / 10) * 100, 100)
#     severity_index = (0.7 * density) + (0.3 * thickness_score)

#     return severity_index, density, thickness


# def severity_label(severity_index):
#     if severity_index < 30:
#         return "Normal"
#     elif severity_index < 60:
#         return "Moderate"
#     else:
#         return "Severe"


# # =========================================
# # MAIN
# # =========================================
# if __name__ == "__main__":

#     vessel_mask = cv2.imread(VESSEL_MASK_PATH, cv2.IMREAD_GRAYSCALE)
#     if vessel_mask is None:
#         raise FileNotFoundError("Vessel mask not found. Run demo_segmentation.py first.")

#     severity_index, density, thickness = calculate_severity_index(vessel_mask)
#     label = severity_label(severity_index)

#     print(f"Vessel Density (%): {density:.2f}")
#     print(f"Average Vessel Thickness: {thickness:.2f}")
#     print(f"Severity Index: {severity_index:.2f}")
#     print(f"Severity Level: {label}")
