import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import LogCosh, Huber
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
import time
import os

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
  'loss': Huber(),
  'encode_activations': [
    'elu',
    'elu',
    'relu'
  ],
  'decode_activations': [
    'relu',
    'elu',
    'elu'
  ],
}

# define the input layer and encoding dimensions of the autoencoder
encoding_dim = 32
input_layer = Input(shape=(train_data.shape[1],))

# create the encoder layers of the autoencoder
encoded = Dense(512, activation='elu')(input_layer)
encoded = Dense(256, activation='elu')(encoded)
encoded = Dense(128, activation=model_settings['encode_activations'][0])(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Dense(64, activation=model_settings['encode_activations'][1])(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Dense(encoding_dim, activation=model_settings['encode_activations'][2])(encoded)

# create the decoder layers of the autoencoder
decoded = Dense(64, activation=model_settings['decode_activations'][0])(encoded)
# decoded = Dropout(rate=0.2)(decoded)
decoded = Dense(128, activation=model_settings['decode_activations'][1])(decoded)
# decoded = Dropout(rate=0.2)(decoded)
decoded = Dense(256, activation='elu')(decoded)
decoded = Dense(512, activation='elu')(decoded)

# define the output layer of the autoencoder
output_layer = Dense(train_data.shape[1], activation=model_settings['decode_activations'][2])(decoded)

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

# train the model
autoencoder.fit(train_data, train_data, 
                epochs=5000, batch_size=200, 
                validation_data=(val_data, val_data),
                callbacks=[lr_scheduler, early_stopping])

# use the trained autoencoder to regenerate the input data
regenerated_data = autoencoder.predict(test_data)

# calculate the mean squared error between the original and regenerated data
mse = mean_squared_error(test_data, regenerated_data)
print("Mean Squared Error: ", mse)

# reset the stdout to the original, and close the custom file
sys.stdout = orig_stdout
output_file.close()

# reroute the stdout to a compararison PNG file
comp_png_file = open(f'../out/{output_id}/comparison.png', 'w')
sys.stdout = comp_png_file

# define plot settings
sample_ind = 10000
max_plot_size = 1000

plt1 = {
  'x': range(min(len(test_data[sample_ind]), max_plot_size)),
  'y': test_data[sample_ind, :min(len(test_data[sample_ind]), max_plot_size)],
  'label': 'Original ECG'
}
plt2 = {
  'x': range(min(len(regenerated_data[sample_ind]), max_plot_size)),
  'y': regenerated_data[sample_ind, :min(len(regenerated_data[sample_ind]), max_plot_size)],
  'label': 'Regenerated ECG'
}

# plot the original data and the regenerated data
plt.plot(plt1['x'], plt1['y'], label=plt1['label'])
plt.plot(plt2['x'], plt2['y'], label=plt2['label'])
plt.legend()
plt.title('Original vs Regenerated ECG')
plt.xlabel('Index')
plt.ylabel('Recording')
plt.savefig(sys.stdout.buffer)
plt.show()

# reset the stdout to the original
comp_png_file.close()
sys.stdout = orig_stdout