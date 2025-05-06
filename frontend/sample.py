import streamlit as st
from PIL import Image
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

model = load_model("../checkpoint/model3_50.keras")

def preprocess(img, size = (224,224)):

    img = np.array(img)
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:  # Already grayscale
        pass
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img,size)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img,axis=0)
    img = np.expand_dims(img, axis=-1) 
    return img

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
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------- Model Inference Button --------
    if st.button("Run Model"):
        st.success("Model ran successfully!")

        # -------- Simulated multi-label output --------
        # Let's say we have 7 classes
        labels = ['Atelectasis','Cardiomegaly','Edema','Effusion','Tortuous Aorta','Calcification of the Aorta','No Finding']
        # preds = np.random.rand(len(labels))  # Random prediction scores
        input_img = preprocess(image)
        preds = model.predict(input_img)[0]

        # Show buttons for classes where score > 0.5
        for i, score in enumerate(preds):
            print(i,score)
            if score > 0.5:
                with st.expander(f"🔍 {labels[i]} (Score: {score:.2f})"):
                    # Simulated Grad-CAM image for now (placeholder)
                    st.image(input_img, caption=f"Grad-CAM for {labels[i]}", use_container_width=True)
                    pass

st.markdown("</div>", unsafe_allow_html=True)
