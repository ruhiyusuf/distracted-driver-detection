from flask import Flask, flash, redirect, render_template, url_for, request

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import tensorflow as tf
import os
import cv2
import urllib.request
import os
from werkzeug.utils import secure_filename
import random
from pathlib import Path


from Driver_Model.ddd_sample_predict import runDDD

#USER NEEDS TO CHANGE THE DIRECTORY PATHS
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['GET', 'POST'])
def upload_image():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        BASE_DIR = Path(__file__).parent.absolute()

        file.save(os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)

        DDDresult = runDDD(filename)
        
        #Takes about 35 seconds after user uploads image

        flash('Image successfully uploaded and displayed below')
        flash('The activity detected: ' + DDDresult)
        # flash('Playlist Link: ', playlist_link)
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()

    # 3000 4000
    # 1024

