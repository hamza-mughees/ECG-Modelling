from scipy.signal import resample
from tensorflow.keras.models import load_model
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import LogCosh, Huber, MeanSquaredError
import time

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

def performance_vis(test_data, model_id, sample_ind, n_segments, overlap, tr_time=None):
    # load the autoencoder model from the directory with model ID
    model = load_model(f'../out/{model_id}/autoencoder.h5')

    # use the trained autoencoder model to regenerate the input on the test data
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

    # loop over each segment and extract the original and regenerated data
    for i in range(n_segments):
        test_sample = test_data[sample_ind+i]
        regenerated_sample = regenerated_data[sample_ind+i]
        # only keep the portion of the data that does not overlap
        test_y.append(test_sample[:int(len(test_sample)*(1-overlap))])
        regenerated_y.append(regenerated_sample[:int(len(regenerated_sample)*(1-overlap))])

    # convert the lists to Numpy arrays
    test_y = np.array(test_y)
    regenerated_y = np.array(regenerated_y)

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

    # plot the original data and the regenerated data
    plt.plot(plt1['x'], plt1['y'], label=plt1['label'])
    plt.plot(plt2['x'], plt2['y'], label=plt2['label'])
    plt.legend()
    plt.title('Original vs Regenerated ECG')
    plt.xlabel('Index')
    plt.ylabel('Recording')
    plt.savefig(sys.stdout.buffer)
    plt.show()

    # reset the stdout to the original
    comp_png_file.close()
    sys.stdout = orig_stdout