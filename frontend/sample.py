import streamlit as st
from PIL import Image
import numpy as np
import random

st.set_page_config(layout="wide")

# -------- Background styling --------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('background.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='centered'>", unsafe_allow_html=True)

st.title("🧠 Multi-Label Classifier of Heart Complications with Grad-CAM")

# -------- Image Upload --------
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------- Model Inference Button --------
    if st.button("Run Model"):
        st.success("Model ran successfully!")

        # -------- Simulated multi-label output --------
        # Let's say we have 7 classes
        labels = ['Atelectasis','Cardiomegaly','Edema','Effusion','Tortuous Aorta','Calcification of the Aorta','No Finding']
        preds = np.random.rand(len(labels))  # Random prediction scores

        # Show buttons for classes where score > 0.5
        for i, score in enumerate(preds):
            if score > 0.1:
                with st.expander(f"🔍 {labels[i]} (Score: {score:.2f})"):
                    # Simulated Grad-CAM image for now (placeholder)
                    st.image("temp1.png", caption=f"Grad-CAM for {labels[i]}", use_column_width=True)

st.markdown("</div>", unsafe_allow_html=True)
