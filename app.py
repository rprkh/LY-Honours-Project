import streamlit as st
import tensorflow as tf
import platform
import numpy as np
import cv2
from PIL import Image
from vit_keras import vit
import tensorflow_addons as tfa
import keras
import matplotlib.pyplot as plt
from options.get_model_prediction import CONFIG, set_seed, make_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten

streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""

st.markdown(streamlit_style, unsafe_allow_html=True)

st.title("**Bird Species Classification Using Vision Transformers**")

def get_system_details_and_devices():
    current_platform = platform.system()
    st.write(f"**Current Platform**: `{current_platform}`  \n**Current Processor**: `{platform.processor()}`  \n")
    if tf.config.experimental.list_physical_devices("GPU"):
        for device in tf.config.experimental.list_physical_devices("GPU"):
            st.write(f"Name: {device.name}, Type: {device.device_type}")
    else:
        pass

    if tf.config.experimental.list_physical_devices("CPU"):
        for device in tf.config.experimental.list_physical_devices("CPU"):
            st.write(f"**Device Name**: `{device.name}`  \n**Device Type**: `{device.device_type}`")
    else:
        pass

def upload_image():
    uploaded_image = st.sidebar.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        file_name = uploaded_image.name
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            file_bytes = np.asarray(bytearray(uploaded_image.read()))
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

            st.image(image_rgb)

            return image_rgb

def initialize_model_architecture(vit_model):
    model = Sequential([
        vit_model,
        Flatten(),
        Dropout(0.2),
        Dense(units = 1024, activation="relu"),
        BatchNormalization(),
        Dense(units = 400, activation="softmax")
    ])

    return model

st.sidebar.title("Trained Models Available")

type_of_model = [
    "ViT_b16",
    "ViT_l32"
]
selected_option = st.sidebar.radio("Select a model:", type_of_model)

get_system_details_and_devices()
set_seed(CONFIG.RANDOM_STATE)
model_input_image = upload_image()

if selected_option == type_of_model[0]:
    if model_input_image is not None:
        vit_model_b16 = vit.vit_b16(image_size=CONFIG.IMAGE_SIZE[0], activation="softmax", 
                            pretrained=True, include_top=False,
                            pretrained_top=False, classes=524)
        vit_model_b16.trainable = False

        vit_b16 = initialize_model_architecture(vit_model_b16)
        vit_b16.load_weights("models/bird_classifier_vit_b16.h5")

        bird_name, confidence = make_predictions(vit_b16, model_input_image)

        st.write(f"**Predicted Bird Species**: {bird_name.title()}  \n**Prediction Confidence**: {round(confidence * 100, 3)}%")

if selected_option == type_of_model[1]:
    vit_model_l32 = vit.vit_l32(image_size=CONFIG.IMAGE_SIZE[0], activation="softmax", 
                        pretrained=True, include_top=False,
                        pretrained_top=False, classes=524)
    vit_model_l32.trainable = False

    vit_l32 = initialize_model_architecture(vit_model_l32)
    vit_l32.load_weights("models/bird_classifier_vit_l32.h5")

    bird_name, confidence = make_predictions(vit_l32, model_input_image)

    st.write(f"**Predicted Bird Species**: {bird_name.title()}  \n**Prediction Confidence**: {round(confidence * 100, 3)}%")
