import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import resample
import csv

subject = '10'   # subject number to load data from, 01-69 (make sure there are two digits)
n_samples = 500

# load the data from the specified file
data = sio.loadmat(f'../ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT/ECGPCG00{subject}.mat')

# extract the ECG data and sampling frequency from the loaded data
raw_ecg = data['ECG'][0]
fs = data["fs"][0, 0]

# print the size of the ECG data and the sampling frequency to the console
print(f'Size: {len(raw_ecg)}')
print(f'Sampling Frequency (fs): {fs}')

# downsample the raw ECG data
fs_new = 250
proc_ecg = resample(raw_ecg, (len(raw_ecg)*fs_new)//fs)

# split the proc_ecg data into 500 arrays of equal length and store it in a list
sample_length = len(proc_ecg) // n_samples
proc_ecg = [proc_ecg[i*sample_length:(i+1)*sample_length] for i in range(n_samples)]

# print the shape of the final dataset that is saved
print(f'Final shape of the dataset: {np.shape(proc_ecg)}')

# save the proc_ecg data in a csv file
with open('../res/data.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  for sample in proc_ecg:
    writer.writerow(sample)

# plot the first segment of the downsampled ECG data
plt.plot(proc_ecg[0])
plt.ylabel('Voltage (mV)')
plt.xlabel('Sample')
plt.title('ECG')
plt.show()  # show the plot