import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import time

sio = socketio.Server(async_mode='eventlet')
telemetry_func = None

def init():
    app = Flask(__name__)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 6603)), app)

@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    telemetry_func(data)

def send_control(angle, torque):
    sio.emit("car", 
        data={
            'angle': angle,
            'torque': torque
        }, 
    skip_sid=True)


if __name__ == "__main__":
    init()

