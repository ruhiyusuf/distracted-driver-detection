from flask import Flask, flash, redirect, render_template, url_for, request, Response

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

BASE_DIR = Path(__file__).parent.parent

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

        DDDresult = runDDD(filename, os.path.join(BASE_DIR, 'static', 'uploads'))
        
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



""" 
def gen(video):
    while True:
        success, image = video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#if __name__ == "__main__":
#    print("Starting the program!")
#    app.run()

    # 3000 4000
    # 1024
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)
"""

if __name__ == "__main__":
    print("Starting the program!")
    app.run()