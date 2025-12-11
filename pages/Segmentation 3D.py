import streamlit as st
from PIL import Image
import requests
import io
import base64
import plotly.graph_objects as go
import plotly.io as pio

import json


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Brain Tumor Segmentation", layout="centered")

# ---------- BACKGROUND COLOR ---------

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        # background-blend-mode: lighten;
        # background-size: cover;
        # background-position: center;
        # background-attachment: fixed;
        background-color: #DCE2F5;
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
        margin-bottom: 10px;
        border-left: 6px solid #fff;
        border-right: 6px solid #fff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    ">
        <h1 style='text-align: center; color: #fff; font-size: 30px; font-weight: bold; font-family: Georgia; margin: 0;'>
            🧠 Brain Tumor Segmentation<br> 3D MRI 
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style="
        background: #D2E5FD;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 22px;
        border-left: 5px solid #22508C;
        border-right: 5px solid #22508C;
        box-shadow: 0 2px 8px rgba(155, 89, 182, 0.08);
    ">
        <p style="font-family: Georgia, sans-serif; font-size: 17px; color: #31125f; font-weight: 500; margin: 0; line-height: 1.55; text-align: left;">
            Our model uses 3D MRI data to detect both the <b>location</b> and the <b>volume</b> of brain tumors.<br>
            The model was trained on the BraTS 2023 dataset, specifically for Glioblastoma multiforme tumors.<br><br>
            This tool is intended to assist doctors with tracking <b>tumor progression over time</b>, and <b>planning surgery</b>.<br><br>
            <b>Disclaimer:</b> This model typically achieves a Dice coefficient of about 78%, meaning it detects at least 80% of tumor's volume.<br>
            In most cases, the predicted tumor closely matches the ground truth, with Dice scores reaching up to 90% (for a very good fit).
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )


# ---------- API URL ----------
#API_URL = "https://brain-782621711539.europe-west1.run.app/predict_classification"
BASE_URL = "https://brain-1042182811091.europe-west1.run.app"
API_URL = f"{BASE_URL}/predict_classification"

# ---------- CLASS LABELS ----------
CLASS_LABELS = {
    0: "   🟢 NO tumor",           
    1: "   🟡 Meningioma tumor",
    2: "   🟡 Glioma tumor",     # Glioblastoma multiforme
    3: "   🟡 Pituitary tumor"
}


# ---------- 3D VISUALIZATION ----------


import pickle
import plotly.graph_objects as go
import streamlit as st

st.title("Visualisation 3D depuis pickle")

# chemin vers le fichier .pkl
pkl_path = "assets/comparaison_tumeur_gt_pred.pkl"  

with open(pkl_path, "rb") as f:
    fig = pickle.load(f)
st.plotly_chart(fig, use_container_width=True)