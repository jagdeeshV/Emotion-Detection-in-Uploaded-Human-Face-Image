# load json and create model
from __future__ import division
import os
import cv2
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from tensorflow.keras.preprocessing import image
import streamlit as st

# Title of the app
st.title("Face Emotion Detection App")

# Load the FER Model Image files & its weights
model = model_from_json(open(r"D:\Guvi\FinalImageProj\yFacial_Expr_Recogn.json",  'r').read())
model.load_weights(r"D:\Guvi\FinalImageProj\fer.h5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 52
HEIGHT = 52
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Instructions
st.write("Upload a Face image to detect the emotion on the face.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#    full_size_image = cv2.imread("test.jpg")
    full_size_image = cv2.imdecode(file_bytes, 1)
    print("Image Loaded")

    #gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    gray=cv2.cvtColor(full_size_image, cv2.COLOR_BGR2GRAY) 

    # Display the uploaded image
    st.image(gray, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting emotion...")

    #detecting faces
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #faces = face.detectMultiScale(gray, 1.3  , 10)
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9) 

    if len(faces) <= 0:
        st.write('Not a face')
    else:
        #Analysing faces
        for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (52, 52)), -1), 0)
                cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                cropped_img = cropped_img / 255.0
                cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #predicting the emotion
                yhat= model.predict(cropped_img)
                cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                print("Emotion: "+labels[int(np.argmax(yhat))])

        cv2.imshow('Emotion', full_size_image)
        cv2.waitKey()

