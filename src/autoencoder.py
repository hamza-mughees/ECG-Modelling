import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, Reshape, AveragePooling1D, Conv1DTranspose, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import LogCosh, Huber, MeanSquaredError
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
import time
import os

import globals
from functions import performance_vis

np.random.seed(1)
tf.random.set_seed(1)

# load the data from the csv file into a pandas dataframe
df = pd.read_csv("../res/allPatients.csv", header=None)

# convert the dataframe to a numpy array
data = df.to_numpy()

# perform the train-test-split
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# perform the train-val-split
train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=False)

# define decay rate and decay operation
def step_decay(epoch, lr):
    decay_rate = 1e-6
    step_size = 1
    if epoch % step_size == 0:
        return lr * (1 - decay_rate)
    return lr

# create learning rate schedular
lr_scheduler = LearningRateScheduler(step_decay)

# create early stopping policy
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# define model settings
model_settings = {
  'optimizer': Adam(learning_rate=0.001),
  'loss': 'mse',
  'encode_activations': [
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'relu'
  ],
  'decode_activations': [
    'relu',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU'
  ],
}

# define the input layer and encoding dimensions of the autoencoder
encoding_dim = 32
input_layer = Input(shape=(train_data.shape[1],))

# create the encoder layers of the autoencoder
encoded = Reshape((train_data.shape[1], 1))(input_layer)
encoded = Conv1D(16, kernel_size=3, padding='same', activation=model_settings['encode_activations'][0])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Conv1D(16, kernel_size=3, padding='same', activation=model_settings['encode_activations'][4])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Conv1D(8, kernel_size=3, padding='same', activation=model_settings['encode_activations'][5])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Conv1D(4, kernel_size=3, padding='same', activation=model_settings['encode_activations'][6])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Conv1D(2, kernel_size=3, padding='same', activation=model_settings['encode_activations'][7])(encoded)

# create the decoder layers of the autoencoder
decoded = Conv1DTranspose(4, kernel_size=3, padding='same', activation=model_settings['decode_activations'][0])(encoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(8, kernel_size=3, padding='same', activation=model_settings['decode_activations'][1])(encoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(16, kernel_size=3, padding='same', activation=model_settings['decode_activations'][2])(encoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(16, kernel_size=3, padding='same', activation=model_settings['decode_activations'][3])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Flatten()(decoded)

# define the output layer of the autoencoder
output_layer = Dense(train_data.shape[1], activation=model_settings['decode_activations'][4])(decoded)

# create the autoencoder model
autoencoder = Model(input_layer, output_layer)

# compile the model
autoencoder.compile(optimizer=model_settings['optimizer'], loss=model_settings['loss'])

# reroute the stdout to a custom output file
output_id = time.strftime('%Y%m%d-%H%M%S')
os.mkdir(f'../out/{output_id}')
output_file_name = f'../out/{output_id}/output.txt'
output_file = open(output_file_name, 'w+')
orig_stdout = sys.stdout
sys.stdout = output_file

# print the model summary
print(autoencoder.summary())

# print the model settings
for key, value in model_settings.items():
  print(f'{key}: {value}')

# get the start time
st = time.time()

# train the model
autoencoder.fit(train_data, train_data, 
                epochs=5000, batch_size=200, 
                validation_data=(val_data, val_data),
                callbacks=[lr_scheduler, early_stopping])

# get the end time
et = time.time()

# save the autoencoder
autoencoder.save(f'../out/{output_id}/autoencoder.h5')

# reset the stdout to the original, and close the custom file
sys.stdout = orig_stdout
output_file.close()

# visualize and save model performance
performance_vis(test_data, model_id=output_id,
                sample_ind=25000, n_segments=10, overlap=globals.overlap, tr_time=et-st)