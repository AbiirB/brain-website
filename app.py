import streamlit as st
from PIL import Image
import requests
import io



st.title("Brain Tumor Classification")

'''
This front queries the model API
'''

#API_URL = "https://brain-782621711539.europe-west1.run.app/predict_classification"
BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/predict_classification"


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
            st.image(image, caption='✅ Uploaded image')
            
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
                    scores = prediction.get("scores")

                    st.header(f"Class prediction: {tumor_class}")
                    st.write("Raw model scores:", scores)

            except requests.exceptions.RequestException as e:
                st.error(f"Error while calling the API: {e}")