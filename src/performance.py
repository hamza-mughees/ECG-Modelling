import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import sys

# identifier for this run of the script
id = '20230206-212920'

# load the autoencoder model from a file
model = load_model(f'../out/{id}/autoencoder.h5')

# load the data from the csv file into a pandas dataframe
df = pd.read_csv("../res/allPatients.csv", header=None)

# convert the dataframe to a numpy array
data = df.to_numpy()

# perform the train-test-split, discarding the training set
_, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# use the trained autoencoder to regenerate the input data
regenerated_data = model.predict(test_data)

# reroute the stdout to a compararison PNG file
orig_stdout = sys.stdout
comp_png_file = open(f'../out/{id}/comparison.png', 'w')
sys.stdout = comp_png_file

# the sample index to start the plot from
sample_ind = 25000
# the number of subjects to plot
n_subjects = 10
# the overlap between the original and regenerated data
overlap = 0.2

# create arrays to store the original and regenerated data
test_y = []
regenerated_y = []

# loop over each subject and extract the original and regenerated data
for i in range(n_subjects):
  test_sample = test_data[sample_ind+i]
  regenerated_sample = regenerated_data[sample_ind+i]
  # only keep the portion of the data that does not overlap
  test_y.append(test_sample[:int(len(test_sample)*(1-overlap))])
  regenerated_y.append(regenerated_sample[:int(len(regenerated_sample)*(1-overlap))])

# convert the lists to Numpy arrays
test_y = np.array(test_y)
regenerated_y = np.array(regenerated_y)

# create dictionaries to store the plot data and labels
plt1 = {
  'x': range(len(test_y.flatten())),
  'y': test_y.flatten(),
  'label': 'Original ECG'
}
plt2 = {
  'x': range(len(regenerated_y.flatten())),
  'y': regenerated_y.flatten(),
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