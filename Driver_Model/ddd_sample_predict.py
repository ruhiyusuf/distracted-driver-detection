import os
from types import prepare_class
from tensorflow.keras import models
import cv2
import numpy as np
from pathlib import Path

def preProcess(x):
    x = x.astype('float64')
    mean_pixel = [103.939, 116.779, 123.68]
    x[0, :, :] -= mean_pixel[0]
    x[1, :, :] -= mean_pixel[1]
    x[2, :, :] -= mean_pixel[2]
    return x

def runDDD(filename, USER_TEST_PATH):
    BASE_DIR = Path(__file__).parent.parent
    print('BASE DIR:', BASE_DIR)
    # BASE_DIR = r'C:\Users\ruhiy\Documents\Machine Learning\Distracted Driver Detection\Distracted Driver Detection Sample Predict'
    MODEL_DIR = os.path.join(BASE_DIR, 'Driver_Model')
    print('model dir:', MODEL_DIR)
    
    MODEL_NAME = 'vgg16' # using the VGG16 model
    MODEL_FILE = 'model_' + MODEL_NAME
    
    MODEL_FILE_H5 = os.path.join(MODEL_DIR, MODEL_FILE + '.h5')
    print(MODEL_FILE_H5)
    
    # MODEL_FILE_JSON = os.path.join(MODEL_DIR, MODEL_FILE + '.json')
    # print(MODEL_FILE_JSON)
    
    # moving this to be an input
    # USER_TEST_PATH = os.path.join(BASE_DIR, 'static', 'uploads')
    
    labels = ["Safe Driving","Texting - Right","Talking on phone - Right","Texting - Left","Talking on phone - Left"
              ,"Operating the Radio", "Drinking", "Reaching Behind", "Hair and Makeup","Talking to Passenger"
              ]

    loaded_model = models.load_model(MODEL_FILE_H5)

    pre_processed_img = cv2.imread(os.path.join(USER_TEST_PATH, filename))

    print('pre type:', type(pre_processed_img))
    # print('pre shape', pre_processed_img.shape)
    # processed_img = cv2.cvtColor(pre_processed_img, cv2.COLOR_BGR2RGB) 
    # print('after converting to rgb')
    # print('af type:', type(processed_img))
    # print('af shape', processed_img.shape)
    processed_img = cv2.resize(pre_processed_img, (224, 224))
    processed_img = np.expand_dims(processed_img, axis=0)
    # processed_img = preProcess(processed_img)
    # // print('post type:', type(processed_img))
    # // print('post shape', processed_img.shape)
    # // print(processed_img)

    yhat = loaded_model.predict(processed_img)

    print(labels[int(np.argmax(yhat))])
    
    return labels[int(np.argmax(yhat))]

    
# runDDD('angry_face.jpg')