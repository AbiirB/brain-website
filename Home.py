import streamlit as st
from PIL import Image
import requests
import io
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Brain Tumor Classification and Segmentation Project", layout="centered")

# ---------- BACKGROUND ----------
# 🔹 Background image 
with open("assets/network.jpg", "rb") as f:   # 👈 put your real path here
    data = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        # background-image: linear-gradient(rgba(255,255,255,0.8), rgba(210,229,253,0.8)), url("data:image/jpg;base64,{data}");
        # background-blend-mode: lighten;
        # background-size: cover;
        # background-position: center;
        # background-attachment: fixed;
        background-color: #111;
    }}
    </style>
    """, unsafe_allow_html=True
)


# ---------- PAGE BANNER ----------
st.markdown(
    """
    <div style="
        background-color: #111;
        padding: 20px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 6px solid #fff;
        border-right: 6px solid #fff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    ">
        <h1 style='text-align: center; color: #fff; font-size: 30px; font-weight: bold; font-family: Georgia; margin: 0;'>
            # 🧠 Brain Tumor Classification and Segmentation
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- BRAIN GIF ----------

# Use st.image for embedding images in Streamlit apps (HTML <img> won't display in public deployments)
st.image(
    "assets/3d-active-brain.gif",
    width='stretch')

# ---------- PROJECT DESCRIPTION ----------

st.markdown(
    """
    <div style="
        background: #BBCAFA;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 22px;
        border-left: 5px solid #4E7AD4;
        border-right: 5px solid #4E7AD4;
        box-shadow: 0 2px 8px rgba(155, 89, 182, 0.08);
    ">
        <p style="font-family: Georgia, sans-serif; font-size: 18px; color: #31125f; font-weight: 500; margin: 0; line-height: 1.55; text-align: left;">
            This project tackles two key tasks related to brain tumor analysis:
            <br><br>
            <b>1. Classification:</b> Determine the type of tumor (Meningioma, Glioma, or Pituitary tumor).<br>
            <b>2. Segmentation / Localization:</b> Identify the location and extent of the tumor in brain MRI scans.
            <br><br>
            Our work uses deep learning models trained on public MRI datasets like <b>BraTS 2023</b> and <b>BRISC</b> dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

