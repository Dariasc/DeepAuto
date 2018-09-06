import argparse
import time

import cv2
import numpy as np

import auto_server
import utils

parser = argparse.ArgumentParser()
parser.add_argument('target', help='when to save the data and stop receiving', type=int)
parser.add_argument('--training-data', help='what training data to load if need be')
parser.add_argument('--log-interval', help='every x telemetry progress will be logged', type=int, default=500)
parser.add_argument('--render', help='whether to render what the AI is seeing', action='store_true')

args = parser.parse_args()

interval = args.log_interval
start = 0
training_data = []
if args.training_data is not None:
    training_data = np.load(args.training_data).tolist()

def telemetry(telemetry):
    img = utils.stringToImg(telemetry["image"])
    aux = telemetry["aux"]
    training_data.append([img, [telemetry['angle'], telemetry['torque'], telemetry['brake']],
        [aux['sensor0'], aux['sensor1'], aux['sensor2'], aux['speed']]])

    if args.render:
        cv2.imshow('render', img)
        cv2.waitKey(10)

    if len(training_data) % interval == 0:
        if len(training_data) == args.target:
            print("Saving training_data...")
            np.save('training_data', training_data)
            return
        print("[{}] Time: {}".format(len(training_data), str(time.time() - start)))

    # Keep the server alive
    auto_server.send_control(0, 0, 0)
    return

start = time.time()
auto_server.telemetry_func = telemetry
auto_server.init()
