import cv2
import os
import datetime
import time
import numpy as np
from pathlib import Path
from threading import Thread

# global variables
stop_thread = False             # controls thread execution
img = None                      # stores the image retrieved by the camera

def start_capture_thread(cap):
    global img, stop_thread

    # continuously read fames from the camera
    while True:
        _, img = cap.read()

        if (stop_thread):
            break


# initialize webcam capture object
# cap = cv2.VideoCapture('http://0.0.0.0:5000/video_feed')
cap = cv2.VideoCapture('http://0.0.0.0:5000/video_feed')

# retrieve properties of the capture object
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap_fps = cap.get(cv2.CAP_PROP_FPS)
fps_sleep = int(1000 / cap_fps)
print('* Capture width:', cap_width)
print('* Capture height:', cap_height)
print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')

# start the capture thread: reads frames from the camera (non-stop) and stores the result in img
t = Thread(target=start_capture_thread, args=(cap,), daemon=True) # a deamon thread is killed when the application exits
t.start()

# initialize time and frame count variables
last_time = datetime.datetime.now()
frames = 0

from Driver_Model.ddd_sample_predict import runDDD
BASE_DIR = Path(__file__).parent.absolute()

while(True):
    # Capture frame-by-frame
    # ret, frame = cap.read()
    frames += 1

    # cv2.imwrite('video_frames/ttf.png', frame)
    # print("\nDone reading image and writing\n\n")
    # time1 = time.time()
    # # x = "hello"
    
    # if counter % framereadrate == 0:
    #     x = runDDD('ttf.png', os.path.join(BASE_DIR, 'video_frames'))
    # frame = cv2.putText(frame,
    # text = x,
    # org = (50, 50),
    # fontFace = cv2.FONT_HERSHEY_DUPLEX,
    # fontScale = 3.0,
    # color = (125, 246, 55),
    # thickness = 3
    # )
    # time2 = time.time()
    # classification_time = np.round(time2-time1, 3)
    # print("Classificaiton Time =", classification_time, "seconds.")

    # measure runtime: current_time - last_time
    delta_time = datetime.datetime.now() - last_time
    elapsed_time = delta_time.total_seconds()

    # compute fps but avoid division by zero
    if (elapsed_time != 0):
        cur_fps = np.around(frames / elapsed_time, 1)

    # TODO: make a copy of the image and process it here if needed
    cv2.imwrite('video_frames/ttf.png', img)
    predict_starttime = time.time()
    x = runDDD('ttf.png', os.path.join(BASE_DIR, 'video_frames'))
    predict_endtime = time.time()

    # draw FPS text and display image
    if (img is not None):
        cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, x, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("webcam", img)

    # wait 1ms for ESC to be pressed
    key = cv2.waitKey(1)
    if (key == 27):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()