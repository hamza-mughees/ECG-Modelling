import numpy as np
import scipy.io as sio
import csv
import os

# import progress_bar and proc_single_ecg functions from functions file
from functions import progress_bar, proc_single_ecg

# data directory containing MAT files
data_dir = '../ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT'

# get a list of all patient files in the directory
files = os.listdir(data_dir)

# loop through all files in the directory
for i in range(len(files)):
    f = files[i]
    mat = sio.loadmat(os.path.join(data_dir, f))
    
    # extract ECG and fs from mat file
    ecg = mat['ECG'][0]
    fs = mat['fs'][0, 0]
    n = len(ecg)
    
    # check if ECG length is 14400000, if not, skip this file
    if n != 14400000:
        continue
    
    # process the ECG data
    ecg_data = proc_single_ecg(ecg, fs, fs_new=250, n_samples=2000, overlap=0.5)
    
    # check if it's the first file, if so set to write mode 'w', else, set to append mode 'a'
    if i == 0:
        mode = 'w'
    else:
        mode = 'a'
    
    with open(f'../res/allPatients.csv', mode, newline='') as f:
        writer = csv.writer(f)
        for sample in ecg_data:
            writer.writerow(sample)
    
    # call the progress_bar function to show progress
    progress_bar(i+1, len(files))