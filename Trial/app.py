import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import cvlib as cv
from tensorflow import keras

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Load your trained models
bmi_model = tf.keras.models.load_model('Trial/best_bmi_model_v3_3.h5', compile=False)
gender_model = tf.keras.models.load_model('Trial/best_gender_model_v1_0.h5')

gender_labels = ["Female","Male"]

def image_resize(image, size):
    # Get the dimensions of the image
    height, width, _ = image.shape
    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Determine the resizing dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    elif aspect_ratio == 1:
        new_width = size
        new_height = size
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    # Resize the image using the determined dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a black canvas of the desired size
    padded_image = np.zeros((size, size, 3), dtype=np.uint8)

    # Calculate the padding values
    pad_top = (size - new_height) // 2
    pad_bottom = size - new_height - pad_top
    pad_left = (size - new_width) // 2
    pad_right = size - new_width - pad_left

    # Copy the resized image onto the canvas with padding
    padded_image[pad_top:pad_top+new_height, pad_left:pad_left+new_width] = resized_image

    return padded_image


def process_face(face):
    image = image_resize(face, size=224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.copy(image)
    image = image / 255.0
    # image = np.expand_dims(image, axis=0)
    return image

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict BMI and gender
def predict(image):
    image = process_face(image)
    bmi_prediction = bmi_model.predict(image)[0][0]  # Assuming single output value for BMI
    gender_prediction = gender_model.predict(image)
    gender_category = np.argmax(gender_prediction)
    return bmi_prediction, gender_category
  


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        im_pil = Image.fromarray(img)
        bmi_prediction, gender_category = predict(im_pil)

        # Display the prediction results
        st.write("BMI Prediction:", bmi_prediction)
        st.write("Gender Category:", gender_labels[gender_category])

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)