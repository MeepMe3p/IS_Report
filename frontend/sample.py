import streamlit as st
from PIL import Image
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("../checkpoint/model3_50.keras")
def grad_cam(model, layer_name,image , class_index, H=224, W=224):

    # turn image to array
# ==============
    # img = tf.keras.utils.load_img(image, color_mode='grayscale', target_size=(H, W))
    # array = tf.keras.utils.img_to_array(img)  # Shape will be (H, W, 1)
    # # array = np.repeat(array, 3, axis=-1)
    # array = np.expand_dims(array, axis=0)    # Add batch dimension: (1, H, W, 1)
    # array = array / 255.0  
    # array = tf.convert_to_tensor(array, dtype=tf.float32)
# ================
    img = image.convert('L').resize((W, H))  # grayscale and resize
    array = tf.keras.utils.img_to_array(img)  # shape (H, W, 1)
    array = np.expand_dims(array, axis=0)  # shape (1, H, W, 1)
    array = array / 255.0
    array = tf.convert_to_tensor(array, dtype=tf.float32)

    print("the shape is: ", array.shape)


    inputs = model.input  # shape (None, 224, 224, 1)

    # Pass through conv2d (1-channel to 3-channel)
    x = model.get_layer("conv2d")(inputs)

    effnet = model.get_layer("efficientnetb0")

    # builds a new functional model that reuses EfficientNet layers
    # create a new model from the conv2d output to:

    cam_layer = effnet.get_layer("block7a_project_bn").output
    final_output = effnet.output

    # from input to CAM + ouputt
    effnet_model = tf.keras.models.Model(inputs=effnet.input, outputs=[cam_layer, final_output])

    # Step 5: Now pass x (output of conv2d) through this
    cam_output, prediction = effnet_model(x)

    # Step 6: Build grad_model for Grad-CAM
    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[cam_output, prediction])

    for i,layers in enumerate(grad_model.layers):
        print(i,layers)
    # grad_model.layers[2].summary()



    with tf.GradientTape() as tape:
        # tape.watch(model.input)
        conv_outputs, predictions = grad_model(array)
        class_score = predictions[:, class_index]

    # Compute the gradient of the class score w.r.t. conv feature map
    grads = tape.gradient(class_score, conv_outputs)

    # Global average pool the gradients to get the weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply weights with feature map channels
    conv_outputs = conv_outputs[0]  # remove batch dim
    cam = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Normalize and resize
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)
    cam = cv2.resize(cam, (W, H))
    cam = 1.0 - cam
    cam = np.uint8(255 * cam)

    original = np.array(img.convert("RGB"))

    # Apply heatmap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return Image.fromarray(overlay)

def compute_gradcam(model, img, y_c,layer_name='top_conv'):
    # preprocessed_input = load_image(img, df)
    # predictions = model.predict(preprocessed_input)

    # print("Loading original image")
    # plt.figure(figsize=(15, 10))
    # plt.subplot(151)
    # plt.title("Original")
    # plt.axis('off')
    # plt.imshow(img[0, :, :, 0], cmap='gray')

    # gradcam = grad_cam(model,layer_name,img, y_c)
    # # plt.subplot(151 + j)
    # plt.axis('off')
    # # plt.imshow(load_image(img,df, preprocess=False),cmap='gray')
    # # plt.imshow(load_image(img,df, preprocess=False),cmap='gray')
    # plt.imshow(gradcam, cmap='jet', alpha = 0.4)
    return grad_cam(model,layer_name,img,y_c)
 


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
            gradcam_img = compute_gradcam(model, image, i)
            # compute_gradcam(model,image,i,"block7a_project_bn")
            # print(i,score)
            if score > 0.5:
                with st.expander(f"🔍 {labels[i]} (Score: {score:.2f})"):
                    # Simulated Grad-CAM image for now (placeholder)
                    st.image(gradcam_img, caption=f"Grad-CAM for {labels[i]}", use_container_width=True)
                    

st.markdown("</div>", unsafe_allow_html=True)
