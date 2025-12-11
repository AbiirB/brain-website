import streamlit as st
from PIL import Image
import requests
import io
import base64


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
            🧠 Brain Tumor Segmentation
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
        <p style="font-family: Georgia, sans-serif; font-size: 18px; color: #31125f; font-weight: 500; margin: 0; line-height: 1.55; text-align: left;">
            This work finds brain tumors' classes using MRI data, with several 2D images, using a deep learning archtecture (DenseNet121) and transfer learning. 
            Model training made use of the BRISC 2025 dataset. 
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


# ---------- FORM ----------
# ---- Classification Page ----
with st.form(key='classification_form'):
    
    uploaded_image = st.file_uploader(' 🔎 Upload an image of a brain MRI scan', type=['jpg', 'jpeg', 'png'])
    submit = st.form_submit_button(" 💡 Make prediction")
    # when user clicks submit button
    if submit:
        if uploaded_image is None:
            st.error("Please upload an image of a brain MRI scan.", icon="🚨")
        else:
            # Read bytes once
            img = uploaded_image.read()
            image = Image.open(io.BytesIO(img))
            
            # Center the uploaded image using columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption='✅ Uploaded image', use_container_width=True)
            
            # Prepare files for API call
            files = {"file": ("image.jpg", img, uploaded_image.type)}
            
            # Call API to make prediction
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code != 200:
                    st.error(f"API error {response.status_code}: {response.text}")
                else:
                    prediction = response.json()
                    tumor_class = prediction.get("class")
                    tumor_label = CLASS_LABELS.get(tumor_class, f"Class {tumor_class}")

                    scores = prediction.get("scores")
                    
                    # Display the prediction in a centered div with a larger font size
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: center;'>
                            <h2 style='font-size:30px; font-weight:bold; color:black; font-family:sans-serif; text-align:center;'>
                                The studied case shows {tumor_label}
                            </h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # ℹ️ Small grey italic disclaimer
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: center;'>   
                            <p style='font-size: 0.8rem; color: grey; font-style: italic;'>
                                Disclaimer: This model provides information with 87% accuracy
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True)
                    

            except requests.exceptions.RequestException as e:
                st.error(f"Error while calling the API: {e}")
                
            