import streamlit as st
import numpy as np
from PIL import Image
from skimage import color, exposure, filters, morphology, transform
from scipy.ndimage import distance_transform_edt

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
    img = transform.resize(img, (512, 512), preserve_range=True).astype(np.uint8)
    gray = color.rgb2gray(img)
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.02)
    return img, enhanced


def segment_vessels(enhanced):
    thresh = filters.threshold_local(enhanced, block_size=15, offset=-0.01)
    vessel_mask = enhanced < thresh
    vessel_mask = morphology.remove_small_objects(vessel_mask, min_size=50)
    vessel_mask = morphology.binary_opening(vessel_mask, morphology.disk(1))
    vessel_mask = morphology.binary_dilation(vessel_mask, morphology.disk(1))
    return vessel_mask


def calculate_metrics(vessel_mask):
    vessel_pixels = np.count_nonzero(vessel_mask)
    total_pixels = vessel_mask.size
    density = (vessel_pixels / total_pixels) * 100

    distance = distance_transform_edt(vessel_mask)
    thickness = 2 * np.mean(distance[vessel_mask])

    thickness_score = min(thickness * 10, 100)
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
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

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
        st.image(original, width=350)

    with col2:
        st.subheader("Enhanced Image")
        st.image(enhanced, clamp=True, width=350)

    with col3:
        st.subheader("Vessel Segmentation")
        st.image(vessel_mask, clamp=True, width=350)

    st.markdown("---")

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Vessel Density (%)", f"{density:.2f}")
    col5.metric("Avg Thickness", f"{thickness:.2f}")
    col6.metric("Severity Index", f"{severity_index:.2f}")
    col7.metric("Severity Level", label)

    st.success("Analysis completed successfully ✅")
else:
    st.info("Please upload a retinal image to begin.")
