import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import resample
import csv
import os

from extra import progress_bar

data_dir = '../ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT'

raw_ecg_data = []
fs_data = []

files = os.listdir(data_dir)
for i in range(len(files)):
  f = files[i]
  mat = sio.loadmat(os.path.join(data_dir, f))
  raw_ecg_data = np.append(raw_ecg_data, mat['ECG'][0])
  fs_data = np.append(fs_data, mat['fs'][0, 0])
  progress_bar(i+1, len(files))
