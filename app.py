import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_classifier_model.h5")
    return model

model = load_model()

# Load class names
@st.cache_data
def load_class_names():
    with open("class_names.txt") as f:
        return [line.strip() for line in f]

class_names = load_class_names()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype("float32")
    image = np.expand_dims(image, axis=0)
    return image


st.title("Plant Disease Detection")
st.write("Upload a leaf image to predict disease")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        preds = model.predict(input_tensor)

        pred_idx = np.argmax(preds)
        confidence = preds[0][pred_idx]

        st.success(f"Prediction: **{class_names[pred_idx]}**")
        st.info(f"Confidence: **{confidence:.2%}**")

