import scipy.io as sio
import matplotlib.pyplot as plt

subject = '10'   # subject number to load data from, 01-69 (make sure there are two digits)

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
proc_ecg = raw_ecg[::fs//fs_new]

# plot the downsampled ECG data
plt.plot(proc_ecg[:2000])
plt.ylabel('Voltage (mV)')
plt.xlabel('Sample')
plt.title('ECG')
plt.show()  # show the plot