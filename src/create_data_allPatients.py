import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import resample
import csv
import os

from extra import progress_bar

# name of the file to save the data
unproc_data = 'unprocessed_allPatients'

# flag to determine if csv file should be created
create = True 

# new sampling frequency
fs_new = 250 

if create:
  data_dir = '../ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT' # directory of .mat files

  ecg_allPatients = [] # list to store ECG data for all patients
  fs_allPatients = [] # list to store sampling frequencies for all patients

  files = os.listdir(data_dir)
  for i in range(len(files)):
    f = files[i]
    mat = sio.loadmat(os.path.join(data_dir, f)) # load .mat file

    ecg = mat['ECG'][0] # ECG data from .mat file
    fs = mat['fs'][0, 0] # sampling frequency from .mat file

    n = len(ecg)
    if n != 14400000:
      continue # if the ECG data is not of the correct length, skip

    # downsample the ECG data to the new frequency
    ds_ecg = resample(ecg, (n*fs_new)//fs) 

    ecg_allPatients.append(ds_ecg) # add the downsampled ECG data to the list
    progress_bar(i+1, len(files)) # display progress
  
  with open(f'../res/{unproc_data}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for patient in ecg_allPatients:
      writer.writerow(patient) # save the data to a csv file

  # df = pd.DataFrame(ecg_allPatients)
  # df.to_csv(f'../res/{unproc_data}.csv', index=False)

df = pd.read_csv(f'../res/{unproc_data}.csv', header=None) # read the csv file

X = df.iloc[:].values # convert dataframe to numpy array

for x in X:
  print(len(x))