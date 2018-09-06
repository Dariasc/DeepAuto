import numpy as np
import cv2

import utils
import auto_server

def telemetry(telemetry):
    print(telemetry['angle'])
    img = utils.stringToImg(telemetry['image'])
    cv2.imshow('car_view', img)
    cv2.waitKey(10)

    auto_server.send_control(0, 0, 0)


auto_server.telemetry_func = telemetry
auto_server.init()
