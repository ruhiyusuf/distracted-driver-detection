import cv2
import os
import datetime
import time
import numpy as np
from pathlib import Path
from threading import Thread

cap = cv2.VideoCapture('my_video.mp4')

# retrieve properties of the capture object
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap_fps = cap.get(cv2.CAP_PROP_FPS)
fps_sleep = int(1000 / cap_fps)
print('* Capture width:', cap_width)
print('* Capture height:', cap_height)
print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')
fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
# fourcc = int(cv2.VideoWriter_fourcc(*'XVID'))
# out = cv2.VideoWriter('my_video_overlay.mp4', fourcc, 20.0, (cap_width, cap_height))
out = cv2.VideoWriter('my_video_overlay.mp4', fourcc, 0.25, (int(cap_width), int(cap_height)))


# initialize time and frame count variables
last_time = datetime.datetime.now()
frames = 0

from Driver_Model.ddd_sample_predict import runDDD
BASE_DIR = Path(__file__).parent.absolute()

# while(True):
while(cap.isOpened()):
    # Capture frame-by-frame
    # ret, frame = cap.read()
    frames += 1

    ret, img = cap.read()
    #if not ret:
    #  print('Reached the end of the video!')
    #  break

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
    filename = 'ttf' + str(frames) + '.png'

    # compute fps but avoid division by zero
    if (elapsed_time != 0):
        cur_fps = np.around(frames / elapsed_time, 1)

    # TODO: make a copy of the image and process it here if needed
    # cv2.imwrite('video_frames/ttf.png', img)
    cv2.imwrite('video_frames/' + filename, img)
    # cv2.imwrite('')
    predict_starttime = time.time()
    x = runDDD(filename, os.path.join(BASE_DIR, 'video_frames'))
    predict_endtime = time.time()
    cv2.putText(img, x, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    out.write(img)
    # draw FPS text and display image
    cv2.imshow('img', img)
    # cv2.imwrite('my_video_overlay', img)
    # if (img is not None):
    #     cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #     cv2.putText(img, x, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #     cv2.imshow('frame', img)

    # wait 1ms for ESC to be pressed
    key = cv2.waitKey(1)
    if (key == 27):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()