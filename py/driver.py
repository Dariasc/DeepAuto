from keras.models import load_model
import auto_server
import base64
import cv2
from PIL import Image
import io
import numpy as np


model = load_model('model-120k-100e.h5')

def telemetry(telemetry):
    img = stringToImg(telemetry['image'])
    cv2.imshow('car_view', img)
    cv2.waitKey(10)
    prediction = model.predict(img.reshape(-1, 120, 80, 1))[0]
    auto_server.send_control(prediction[0].item(), prediction[1].item())

def stringToImg(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)


auto_server.telemetry_func = telemetry
auto_server.init()