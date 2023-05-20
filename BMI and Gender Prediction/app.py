import cv2
import numpy as np
from keras.models import load_model
from keras_vggface.utils import preprocess_input
import cvlib as cv
import tensorflow as tf
from tensorflow import keras

model = load_model('bmi_model.h5',compile=False)
gender_model = load_model('gender_model.h5')

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

import concurrent.futures
from PIL import Image
# Define function to process each face
def process_face(face):
    image = image_resize(face, size=224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.copy(image)
    image = image / 255.0
    # image = np.expand_dims(image, axis=0)
    return image

cap = cv2.VideoCapture(0)
gender_labels = ["Female","Male"]
while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using cvlib
    faces, _ = cv.detect_face(frame)
    # print(faces)

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
    
    # face_images = [frame[y-100:y2+100, x-100:x2+100] for (x, y, x2, y2) in faces]
    # cv2.imshow("face",face_images[0])
    # Process faces in parallel
    if len(face_images)>0:
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     results = executor.map(lambda face: process_face(face), face_images)
        face_images = np.array(face_images,dtype="float32")
        # print(face_images.shape)    
        prediction = model.predict(face_images,batch_size=32)
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


    # Display the frame with bounding boxes and BMI predictions
    cv2.imshow('BMI Prediction', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()