from scipy.signal import resample
from keras.models import load_model
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import LogCosh, Huber, MeanSquaredError
import time
import os

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

def proc_single_ecg(raw_ecg, fs, fs_new, n_samples, overlap=0, save_dir=None):
    # downsample the raw ECG data
    ds_ecg = resample(raw_ecg, (len(raw_ecg)*fs_new)//fs)

    # save downsampled ECG to folder, if given
    if save_dir:
        save_path = f'../res/{save_dir}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}/output.txt', 'w') as f:
            for v in ds_ecg:
                f.write(f'{v}\n')

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
        for i in range(15):
            regenerations.append(model.predict(test_data))
        regenerated_data = np.mean(regenerations, axis=0)
        regenerated_std = np.std(regenerations, axis=0)
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
    
    # reroute the stdout to a comparison PNG file
    orig_stdout = sys.stdout
    comp_png_file = open(f'../out/{model_id}/comparison.png', 'w')
    sys.stdout = comp_png_file

    # plot the original data and the regenerated data (2d)
    test_ecg = plt1['y']
    regenerated_ecg = plt2['y']
    plot_comparison(test_ecg, regenerated_ecg)

    # close the comparison file
    comp_png_file.close()

    if bayes:
        # reroute the stdout to a gaussians PNG file
        comp_png_file = open(f'../out/{model_id}/gaussians.png', 'w')
        sys.stdout = comp_png_file

        # plot the regenerated means with Gaussian distributions (3d)
        test_ecg = plt1['y']
        means = plt2['y']
        stds = plt2['err']
        plot_gaussians_3d(means, stds, test_ecg=test_ecg)

        # close the Gaussians file
        comp_png_file.close()
    
    # reset the stdout to the original
    sys.stdout = orig_stdout

def plot_comparison(test_ecg, regenerated_ecg):
    # plot the original data and the regenerated data
    plt.plot(np.arange(len(test_ecg)), test_ecg, label='Original ECG')
    plt.plot(np.arange(len(regenerated_ecg)), regenerated_ecg, label='Regenerated ECG')
    plt.legend()
    plt.title('Original vs Regenerated ECG')
    plt.xlabel('Index')
    plt.ylabel('Recording')
    plt.savefig(sys.stdout.buffer)
    plt.show()

def plot_gaussians_3d(means, stds, test_ecg=None, threshold=0.01):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-0.5, 0.5, 1000) # 1000 points between -0.5 and 0.5

    fig2, ax2 = plt.subplots()  # Create a new figure and axis for the bird's eye view plot

    # loop over means and stds to plot each gaussian
    for i, (mean, std) in enumerate(zip(means, stds)):
        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        # find the indices of the values in y that are above the threshold
        idx = np.where(y > threshold)[0]
        # create new arrays that only include the relevant parts of the distribution
        x_new = x[idx]
        y_new = y[idx]
        z_new = i * np.ones_like(x_new)
        ax.plot(z_new, x_new, y_new, 'b-', alpha=0.2)

        # Add the shaded interval between the two ends of the distribution to the bird's eye view plot
        if len(x_new) > 0:
            lower_bound = x_new[0]
            upper_bound = x_new[-1]
            ax2.fill_between([i, i], lower_bound, upper_bound, alpha=0.2, color='b')

    # plot the means and the test data if given
    ax.plot(np.arange(len(means)), means, np.zeros_like(means), 'r-', linewidth=0.5, label='Regenerated ECG Means')

    ax.set_xlabel('Index')
    ax.set_ylabel('Recording')
    ax.set_zlabel('Probability Density')
    ax.set_title('Regenerated ECG Means with Gaussian Distributions')
    ax.legend()
    # plt.show()

    # Customize the bird's eye view plot
    ax2.plot(np.arange(len(test_ecg)), test_ecg, 'g-', linewidth=0.5, label='Original ECG')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Recording')
    ax2.set_title('Original ECG with Regenerated Gaussian Intervals')
    ax2.legend()
    plt.show()

def loss_single_ecg(model_id, ecg_data):
    # load the autoencoder model from the directory with model ID
    model = load_model(f'../out/{model_id}/autoencoder.h5')

    # regenerate the ECG samples using the autoencoder model
    regenerations = []
    for _ in range(15):
        regenerations.append(model.predict(ecg_data))
    regen_ecg = np.mean(regenerations, axis=0)

    # compute the mean squared error between the original and reconstructed ECG samples
    losses = []
    for orig, regen in zip(ecg_data, regen_ecg):
        losses.append(MeanSquaredError()(orig, regen).numpy())
    
    # plot the losses
    plt.plot(range(len(losses)), losses, '-r', linewidth=2)
    plt.xlabel('ECG Sample')
    plt.ylabel('Loss')
    plt.title('Losses over ECG Samples (Autoencoder)')
    plt.xlim(1, 2600)
    plt.ylim(0, 0.2)
    plt.show()

    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    std_loss = np.std(losses)
    min_loss = np.min(losses)
    max_loss = np.max(losses)

    print(f"Mean loss: {mean_loss:.4f}")
    print(f"Median loss: {median_loss:.4f}")
    print(f"Standard deviation of loss: {std_loss:.4f}")
    print(f"Minimum loss: {min_loss:.4f}")
    print(f"Maximum loss: {max_loss:.4f}")
