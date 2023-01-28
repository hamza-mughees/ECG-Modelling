import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import resample
import csv
import os

from functions import progress_bar, proc_single_ecg

data_dir = '../ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT'
files = os.listdir(data_dir)

for i in range(len(files)):
  f = files[i]
  mat = sio.loadmat(os.path.join(data_dir, f))

  ecg = mat['ECG'][0]
  fs = mat['fs'][0, 0]

  n = len(ecg)
  if n != 14400000:
    continue

  ecg_data = proc_single_ecg(ecg, fs, fs_new=250, n_samples=1000, overlap=0)

  if i == 0:
    mode = 'w'
  else:
    mode = 'a'
  
  with open(f'../res/allPatients.csv', mode, newline='') as f:
    writer = csv.writer(f)
    for sample in ecg_data:
      writer.writerow(sample)
  
  progress_bar(i+1, len(files))