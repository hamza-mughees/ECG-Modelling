from scipy.signal import resample
from keras.models import load_model
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import LogCosh, Huber, MeanSquaredError
import time
import tensorflow as tf
import io
from mpl_toolkits.mplot3d import Axes3D

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    # calculate the percentage of completion
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    
    # calculate the filled length of the progress bar
    filled_length = int(length * iteration // total)
    
    # generate the progress bar
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # print the progress bar
    print(f'\r{prefix} {iteration}/{total} |{bar}| {percent}% {suffix}', end='\r')
    
    # print new line on complete
    if iteration == total: 
        print()

def proc_single_ecg(raw_ecg, fs, fs_new, n_samples, overlap=0):
    # downsample the raw ECG data
    ds_ecg = resample(raw_ecg, (len(raw_ecg)*fs_new)//fs)

    # calculate window size for overlapping segments
    window_size = int((len(ds_ecg) // n_samples) // (1-overlap))
    ecg_data = []

    # create the samples for the ecg using the overlapping windows
    for i in range(0, len(ds_ecg) - int(window_size*(1-overlap)), int(window_size*(1-overlap))):
        ecg_data.append(ds_ecg[i:i+window_size])
    
    return ecg_data

def performance_vis(test_data, model_id, sample_ind, n_segments, overlap, tr_time=None, bayes=False):
    # load the autoencoder model from the directory with model ID
    model = load_model(f'../out/{model_id}/autoencoder.h5')

    if bayes:
        # if bayes is true, use the model to regenerate the input data multiple times
        # and calculate the mean and standard deviation of the regenerations
        regenerations = []
        for i in range(5):
            regenerations.append(model.predict(test_data))
        regenerated_data = np.mean(regenerations, axis=0)
        regenerated_std = np.std(regenerations, axis=0)
        # print(f'Means: {regenerated_data[0]}')
        # print(f'Standard Deviations: {regenerated_std[0]}')
    else:
        # use the model to regenerate the input data once
        regenerated_data = model.predict(test_data)

    if tr_time:
        orig_stdout = sys.stdout
        with open(f'../out/{model_id}/output.txt', 'a') as f:
            sys.stdout = f

            # print the time elapsed during training
            print(f'Training time: {time.strftime("%H:%M:%S", time.gmtime(tr_time))} secs')

            # calculate performance metrics between the original and regenerated data
            mse = MeanSquaredError()(test_data, regenerated_data).numpy()
            print(f'Mean Squared Error: {mse}')
            lc = LogCosh()(test_data, regenerated_data).numpy()
            print(f'Log Cosh: {lc}')
            h = Huber()(test_data, regenerated_data).numpy()
            print(f'Huber: {h}')

            sys.stdout = orig_stdout
    
    # reroute the stdout to a compararison PNG file
    orig_stdout = sys.stdout
    comp_png_file = open(f'../out/{model_id}/comparison.png', 'w')
    sys.stdout = comp_png_file

    # create arrays to store the original and regenerated data
    test_y = []
    regenerated_y = []

    if bayes:
        regenerated_err = []

    # loop over each segment and extract the original and regenerated data
    for i in range(n_segments):
        test_sample = test_data[sample_ind+i]
        regenerated_sample = regenerated_data[sample_ind+i]
        # only keep the portion of the data that does not overlap
        test_y.append(test_sample[:int(len(test_sample)*(1-overlap))])
        regenerated_y.append(regenerated_sample[:int(len(regenerated_sample)*(1-overlap))])
        # repeat for bayes if bayes true
        if bayes:
            regenerated_std_sample = regenerated_std[sample_ind+i]
            regenerated_err.append(regenerated_std_sample[:int(len(regenerated_std_sample)*(1-overlap))])

    # convert the lists to Numpy arrays
    test_y = np.array(test_y)
    regenerated_y = np.array(regenerated_y)
    # repeat for bayes if bayes is true
    if bayes:
        regenerated_err = np.array(regenerated_err)

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

    # add error if bayes is true
    if bayes:
        plt2['err'] = regenerated_err.flatten()

    if not bayes:
        # plot the original data and the regenerated data
        plt.plot(plt1['x'], plt1['y'], label=plt1['label'])
        plt.plot(plt2['x'], plt2['y'], label=plt2['label'])
        plt.legend()
        plt.title('Original vs Regenerated ECG')
        plt.xlabel('Index')
        plt.ylabel('Recording')
        plt.savefig(sys.stdout.buffer)
        plt.show()
    else:
        means = plt2['y']
        stds = plt2['err']

        plot_gaussians_3d(means, stds, sys.stdout.buffer)

    # reset the stdout to the original
    comp_png_file.close()
    sys.stdout = orig_stdout

def plot_gaussians_3d(means, stds, buffer):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(min(means), max(means), 1000) # 1000 points between -5 and 5

    # Loop over means and stds to plot each gaussian
    for i, (mean, std) in enumerate(zip(means, stds)):
        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax.plot(i*np.ones_like(x), x, y, 'b-', alpha=0.1)

    ax.set_xlabel('Index')
    ax.set_ylabel('Recording')
    ax.set_zlabel('Probability Density')
    plt.savefig(buffer)
    plt.show()