import auto_server
from PIL import Image
import cv2
import base64
import io
import numpy as np
import time
import os

start = 0
training_data = []
if os.path.isfile('training_data.npy'):
    training_data = np.load('training_data.npy').tolist()

save_interval = False
target_length = 120000

def telemetry(telemetry):
    img = stringToImg(telemetry["image"])
    training_data.append([img, [telemetry['angle'], telemetry['torque']]])

    #cv2.imshow('image', img)
    #cv2.waitKey(10)

    if len(training_data) % 500 == 0:
        if save_interval or len(training_data) == target_length:
            np.save('training_data', training_data)
        print("[{}] Time: {}".format(len(training_data), str(time.time() - start)))
    
    auto_server.send_control(0, 0)
    return

def stringToImg(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

start = time.time()
auto_server.telemetry_func = telemetry
auto_server.init()