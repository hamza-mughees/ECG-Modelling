import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import resample
import csv

subject = '10' # subject number to load data from, 01-69 (make sure there are two digits)
fs_new = 250 # new sampling frequency
n_samples = 500 # number of samples in the final dataset
overlap = 0.5 # overlap percentage between windows

# load the data from the specified file
data = sio.loadmat(f'../ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT/ECGPCG00{subject}.mat')

# extract the ECG data and sampling frequency from the loaded data
raw_ecg = data['ECG'][0]
fs = data["fs"][0, 0]

# print the size of the ECG data and the sampling frequency to the console
print(f'Size: {len(raw_ecg)}')
print(f'Sampling Frequency (fs): {fs}')

# downsample the raw ECG data
ds_ecg = resample(raw_ecg, (len(raw_ecg)*fs_new)//fs)

# calculate window size for overlapping segments
window_size = int((len(ds_ecg) // n_samples) // overlap)
ecg_data = []

# create the dataset using the overlapping windows
for i in range(0, len(ds_ecg) - int(window_size*overlap), int(window_size*overlap)):
  ecg_data.append(ds_ecg[i:i+window_size])

# print the shape of the final dataset that is saved
print(f'Final shape of the dataset: {np.shape(ecg_data)}')

# save the proc_ecg data in a csv file
with open('../res/data.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  for sample in ecg_data:
    writer.writerow(sample)

# plot the first 1000 points of the first segment of the downsampled ECG data
plt.plot(ecg_data[0, :min(len(ecg_data[0], 1000))])
plt.ylabel('Voltage (mV)')
plt.xlabel('Sample')
plt.title('ECG')
plt.show()  # show the plot