# Keras
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

# Set GPU VRam usage to only 30%
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('training-data', help='training data file to use')
parser.add_argument('--epochs', help='how many epochs to run', type=int, default=20)
parser.add_argument('--tensorboard', help='save run to tensorboard', action='store_true')

args = parser.parse_args()


training_data = np.load(args.training-data)

TRAIN_SIZE = len(training_data) / 1000
EPOCHS = args.epochs

INPUT_SHAPE = (120, 80, 1)
LR = 1.0e-4
MODEL_NAME = 'model-{}k-{}e.h5'.format(str(TRAIN_SIZE), str(EPOCHS))

x_train = np.array([i[0] for i in training_data]).reshape(-1, 120, 80, 1)
y_train = np.array([i[1] for i in training_data])

def build_model():
    """
    https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py#L44

    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(2))
    model.summary()

    return model

def train_model(model):
    fit_callbacks = []
    if args.tensorboard:
        logdir = './tensorboard/autoAuto/{}k/{}-epochs'.format(str(TRAIN_SIZE), str(EPOCHS))
        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                                write_graph=True)

        fit_callbacks.append(tensorboard)

    model.compile(loss='mean_squared_error', optimizer=Adam(LR), metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.05, callbacks=fit_callbacks)
    model.save(MODEL_NAME)

train_model(build_model())
