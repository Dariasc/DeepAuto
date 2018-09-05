# Keras
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, concatenate
from keras.models import Model

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

INPUT_SHAPE = (180, 120, 1)

LR = 1.0e-4
MODEL_NAME = 'model-v0.2-{}k-{}e.h5'.format(str(TRAIN_SIZE), str(EPOCHS))

cnn_input = np.array([i[0] for i in training_data]).reshape(-1, 180, 120, 1)
aux_input = np.array([i[2] for i in training_data])
labels = np.array([i[1] for i in training_data])

def build_model():
    """
    Utilize Nvidia Model

    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    main_input = Input(shape=INPUT_SHAPE, name="cnn_input")
    cnn = Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)(main_input)
    cnn = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(cnn)
    cnn = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(cnn)
    cnn = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2))(cnn)
    cnn = Conv2D(64, 3, 3, activation='elu')(cnn)
    cnn = Conv2D(64, 3, 3, activation='elu')(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(100, activation='elu')(cnn)
    cnn = Dense(50, activation='elu')(cnn)
    cnn = Dense(10, activation='elu')(cnn)

    cnn_output = Dense(3, name='cnn_output')(cnn)

    aux = Input(shape=(4,), name='aux_input')

    merge = concatenate([cnn, aux])
    merge = Dense(10)(merge)
    merge = Dense(5)(merge)

    output = Dense(3, name='output')(merge)

    model = Model(inputs=[main_input, aux], outputs=[output, cnn_output])
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

    model.fit([cnn_input, aux_input], [labels, labels], epochs=EPOCHS, validation_split=0.05, callbacks=fit_callbacks)
    model.save(MODEL_NAME)

train_model(build_model())
