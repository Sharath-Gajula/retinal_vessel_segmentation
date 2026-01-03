import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Retinal Vessel Severity Analysis",
    layout="wide"
)

st.title("🩺 Retinal Vessel Segmentation & Severity Analysis")
st.write("Upload a retinal fundus image to analyze vessel severity.")

# =========================================
# FUNCTIONS
# =========================================
def preprocess_image(img):
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return img, enhanced


def segment_vessels(enhanced):
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

    return vessel_mask


def calculate_metrics(vessel_mask):
    vessel_pixels = np.count_nonzero(vessel_mask)
    total_pixels = vessel_mask.size
    density = (vessel_pixels / total_pixels) * 100

    binary = np.where(vessel_mask > 0, 255, 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thickness = 2 * np.mean(dist_transform[binary == 255])

    thickness_score = min((thickness / 10) * 100, 100)
    severity_index = (0.7 * density) + (0.3 * thickness_score)

    return density, thickness, severity_index


def severity_label(severity_index):
    if severity_index < 30:
        return "Normal"
    elif severity_index < 60:
        return "Moderate"
    else:
        return "Severe"


# =========================================
# FILE UPLOAD
# =========================================
uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "png", "jpeg", "tif"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Process image
    original, enhanced = preprocess_image(img)
    vessel_mask = segment_vessels(enhanced)

    density, thickness, severity_index = calculate_metrics(vessel_mask)
    label = severity_label(severity_index)

    # =========================================
    # DISPLAY RESULTS
    # =========================================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.subheader("Enhanced Image (CLAHE)")
        st.image(enhanced, clamp=True, use_column_width=True)

    with col3:
        st.subheader("Vessel Segmentation")
        st.image(vessel_mask, clamp=True, use_column_width=True)

    st.markdown("---")

    # =========================================
    # METRICS
    # =========================================
    col4, col5, col6, col7 = st.columns(4)

    col4.metric("Vessel Density (%)", f"{density:.2f}")
    col5.metric("Avg Thickness", f"{thickness:.2f}")
    col6.metric("Severity Index", f"{severity_index:.2f}")
    col7.metric("Severity Level", label)

    st.success("Analysis completed successfully ✅")

else:
    st.info("Please upload a retinal image to begin.")
