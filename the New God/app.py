import streamlit as st
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,WebRtcMode
import av
import cv2
import numpy as np
import cvlib as cv
from PIL import Image
from keras.models import load_model
st.title("Test")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
model = load_model('the New God/best_bmi_model_v3_3.h5',compile=False)
age_model = load_model('the New God/best_age_model_v1_2.h5',compile=False)
gender_model = load_model('the New God/best_gender_model_v1_0.h5')


# webrtc_streamer(key="Sample")

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
    pad_left = (size - new_width) // 2

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

def process_box(frame,f):
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
    return face_select

class VideoProcessor:
    def recv(self,frame):
        gender_labels = ["Female","Male"]
        print("Started")
        img = frame.to_ndarray(format="bgr24")
    # Detect faces in the frame using cvlib
        faces,_ = cv.detect_face(img)
        # print(faces)
        # print(faces)
        
        # Prepare faces for parallel processing
        print("1")
        
        face_images = []
        for idx,i in enumerate(faces):
            x,y,w,h = i
            cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
            face_images.append(process_box(img,i))
            # face_images = [frame[y-100:y2+100, x-100:x2+100] for (x, y, x2, y2) in faces]
        try:
            if len(face_images)>0:
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     results = executor.map(lambda face: process_face(face), face_images)
                face_images = np.array(face_images,dtype="float32")
                # print(face_images.shape)    
                # print(face_images.shape)
                prediction = model.predict(face_images,batch_size=32)
                gender_prediction = gender_model.predict(face_images,batch_size=32)
                age_prediction = age_model.predict(face_images, batch_size=32)
                # print(prediction)
                # print(gender_prediction)
                # # Draw bounding boxes and display BMI for each face
                for (x, y, x2, y2), bmi_value,gender_pred, age_pred in zip(faces, prediction,gender_prediction, age_prediction):
                    print(bmi_value[0])
                    gender_pred_label = gender_labels[np.argmax(gender_pred)]
                    print(gender_pred_label)
                    print(round(age_pred[0]))
                    # cv2.putText(frame, f'BMI: {bmi_value[0]:.2f}  Gender: {gender_pred_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(img, f'BMI: {bmi_value[0]:.2f}', (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(img, f'Gender: {gender_pred_label}', (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(img, f'Age: {bmi_value[0]}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print(e)
        # 
        # cv2.imshow("face",face_images[0])
        # Process faces in parallel
        # 
        return av.VideoFrame.from_ndarray(img,format="bgr24")
    
webrtc_streamer(key="edge",mode=WebRtcMode.SENDRECV,video_processor_factory=VideoProcessor,rtc_configuration=RTC_CONFIGURATION,media_stream_constraints={"video": True, "audio": False})