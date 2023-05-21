import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import cvlib as cv
from tensorflow import keras
import urllib.request

# Load your trained models
bmi_model = tf.keras.models.load_model('BMI/best_bmi_model_v3_3.h5', compile=False)
gender_model = tf.keras.models.load_model('BMI/best_gender_model_v1_0.h5')

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
    image = preprocess_image(image)
    bmi_prediction = bmi_model.predict(image)[0][0]  # Assuming single output value for BMI
    gender_prediction = gender_model.predict(image)
    gender_category = np.argmax(gender_prediction)
    return bmi_prediction, gender_category

# Streamlit app
def main():
    st.title("BMI and Gender Prediction")
    st.write("Upload a photo or use your webcam to predict BMI and gender.")

    # Choose between webcam or upload photo
    option = st.sidebar.selectbox("Select an option:", ("Webcam", "Upload Photo"))

    if option == "Webcam":
        st.write("Webcam")
        run_webcam()

    if option == "Upload Photo":
        st.write("Upload Photo")
        run_upload_photo()

# Function to run the webcam
# Function to run the webcam
def run_webcam():
    video_capture = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    cnt = 0
    while True:
        cnt += 1
        _, frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FRAME_WINDOW.image(frame_rgb, channels="RGB", use_column_width=True)

        faces, _ = cv.detect_face(frame)

        face_images = []
        # Prepare faces for parallel processing
        for idx, f in enumerate(faces):
            x, y = f[0], f[1]
            x2, y2 = f[2], f[3]
            x = x-100
            x2 = x2+100
            y = y-100
            y2 = y2+100
            x = 0 if x<1 else x
            y = 0 if y<1 else y
            face_select = frame[y:y2,x:x2]
            face_select = process_face(face_select)
            face_images.append(face_select)

        if len(face_images)>0:
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     results = executor.map(lambda face: process_face(face), face_images)
            face_images = np.array(face_images,dtype="float32")
            # print(face_images.shape)    
            prediction = bmi_model.predict(face_images,batch_size=32)
            gender_prediction = gender_model.predict(face_images,batch_size=32)
            
            # Draw bounding boxes and display BMI for each face
            for (x, y, x2, y2), bmi_value,gender_pred in zip(faces, prediction,gender_prediction):
                print(bmi_value[0])
                
                gender_pred_label = gender_labels[np.argmax(gender_pred)]
                print(gender_pred_label)
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(frame, f'BMI: {bmi_value[0]:.2f}  Gender: {gender_pred_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, f'BMI: {bmi_value[0]:.2f}', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, f'Gender: {gender_pred_label}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # FRAME_WINDOW.image(frame_rgb)
        # Display the frame with bounding boxes and BMI predictions
        # cv2.imshow('BMI Prediction', frame)
        # FRAME_WINDOW.image(frame, channels="RGB", use_column_width=True)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", use_column_width=True)
            # predict_button_clicked = False
            
# Function to upload and predict with a photo
def run_upload_photo():
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')

        if st.button("Predict", key='photo_button'):
            bmi_prediction, gender_category = predict(image)

            st.write("BMI Prediction:", bmi_prediction)
            st.write("Gender:", gender_category)

if __name__ == '__main__':
    main()
