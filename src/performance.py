import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import sys

import globals
from functions import performance_vis

# load the data from the csv file into a pandas dataframe
df = pd.read_csv("../res/allPatients.csv", header=None)

# convert the dataframe to a numpy array
data = df.to_numpy()

# perform the train-test-split, discarding the training set
_, test_data = train_test_split(data, test_size=0.2, shuffle=False)

performance_vis(test_data, model_id='20230206-212920',
                sample_ind=25000, n_segments=10, overlap=globals.overlap)