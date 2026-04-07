import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from PIL import Image

# Load model
model = load_model("brain_90.h5")

# Class order
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Prediction", "Model Details"])

# =========================
# 🔹 PAGE 1: PREDICTION
# =========================
if page == "Prediction":
    st.title("🧠 Brain Tumor Detection")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)[0]

        st.subheader("Raw Output:")
        st.write(prediction)

        st.subheader("Class-wise Probabilities:")
        for i in range(4):
            st.write(f"{class_names[i]}: {prediction[i]:.4f}")

        class_index = np.argmax(prediction)
        st.success(f"Prediction: {class_names[class_index]}")

# =========================
# 🔹 PAGE 2: MODEL DETAILS
# =========================
elif page == "Model Details":
    st.title("📊 Model Information")

    st.header("🧠 Model Overview")
    st.write("""
    - Model: ResNet50V2 (Transfer Learning)
    - Type: Multi-class Classification
    - Classes: Glioma, Meningioma, No Tumor, Pituitary
    """)

    st.header("⚙️ Training Details")
    st.write("""
    - Pretrained on ImageNet
    - Fine-tuned on Brain MRI dataset
    - Input Size: 224 x 224 x 3
    - Architecture: CNN with custom classifier
    """)

    st.header("📈 Performance")
    st.write("""
    - Validation Accuracy: 91%
    - Validation Loss: 0.26
    """)

    st.header("🛠 Technologies Used")
    st.write("""
    - TensorFlow / Keras
    - Python
    - Streamlit
    """)

    st.header("📌 Notes")
    st.write("""
    - Model uses preprocessing specific to ResNet50V2
    - Predictions are probability-based (Softmax output)
    - Suitable for academic and demo purposes
    """)