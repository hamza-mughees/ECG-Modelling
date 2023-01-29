import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import resample
import csv

from functions import proc_single_ecg

subject = '10' # subject number to load data from, 01-69 (make sure there are two digits)
fs_new = 250 # new sampling frequency
n_samples = 1000 # number of samples in the final dataset
overlap = 0.5 # overlap percentage between windows

# load the data from the specified file
data = sio.loadmat(f'../ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT/ECGPCG00{subject}.mat')

# extract the ECG data and sampling frequency from the loaded data
raw_ecg = data['ECG'][0]
fs = data["fs"][0, 0]

# print the size of the ECG data and the sampling frequency to the console
print(f'Size: {len(raw_ecg)}')
print(f'Sampling Frequency (fs): {fs}')

# process the ECG data
ecg_data = proc_single_ecg(raw_ecg, fs, fs_new=250, n_samples=1000, overlap=0.5)

# print the shape of the final dataset that is saved
print(f'Final shape of the dataset: {np.shape(ecg_data)}')

# save the proc_ecg data in a csv file
with open('../res/singlePatient.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  for sample in ecg_data:
    writer.writerow(sample)

# plot the first 1000 points of the first segment of the downsampled ECG data
plt.plot(ecg_data[0][:min(len(ecg_data[0]), 1000)])
plt.ylabel('Voltage (mV)')
plt.xlabel('Sample')
plt.title('ECG')
plt.show()  # show the plot