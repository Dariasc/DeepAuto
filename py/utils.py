from PIL import Image
import io
import cv2
import base64
import numpy as np

def stringToImg(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)