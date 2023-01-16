import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
import time
import os

# load the data from the csv file into a pandas dataframe
df = pd.read_csv("../res/data.csv", header=None)

# convert the dataframe to a numpy array
data = df.to_numpy()

# perform the train-test-split
train_data, test_data = train_test_split(data, test_size=0.2)

# define model settings
model_settings = {
  'optimizer': Adam(lr=0.001, decay=1e-6),
  'loss': 'mse',
}

# define the input and encoding dimensions of the autoencoder
encoding_dim = 32
input_data = Input(shape=(train_data.shape[1],))

# create the encoder layers of the autoencoder
encoded = Dense(128, activation='relu')(input_data)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# create the decoder layers of the autoencoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(train_data.shape[1], activation='sigmoid')(decoded)

# create the autoencoder model
autoencoder = Model(input_data, decoded)

# compile the model
autoencoder.compile(optimizer=model_settings['optimizer'], loss=model_settings['loss'])

# reroute the stdout to a custom output file
output_id = int(time.time())
os.mkdir(f'../out/{output_id}')
output_file_name = f'../out/{output_id}/output.txt'
output_file = open(output_file_name, 'w+')
orig_stdout = sys.stdout
sys.stdout = output_file

# print the model summary
print(autoencoder.summary())

# train the model
autoencoder.fit(train_data, train_data, epochs=130, batch_size=4)

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
sample_ind = 1

# plot the original data and the regenerated data
plt.plot(range(len(test_data[sample_ind])), test_data[sample_ind], label='Original Data')
plt.plot(range(len(regenerated_data[sample_ind])), regenerated_data[sample_ind], label='Regenerated Data')
plt.legend()
plt.title('Original vs Regenerated ECG')
plt.xlabel('Index')
plt.ylabel('Recording')
plt.show()
plt.savefig(sys.stdout.buffer)

# reset the stdout to the original
comp_png_file.close()
sys.stdout = orig_stdout