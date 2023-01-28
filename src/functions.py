from scipy.signal import resample

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} {iteration}/{total} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def proc_single_ecg(raw_ecg, fs, fs_new, n_samples, overlap):
    # downsample the raw ECG data
    ds_ecg = resample(raw_ecg, (len(raw_ecg)*fs_new)//fs)

    # calculate window size for overlapping segments
    window_size = int((len(ds_ecg) // n_samples) // (1-overlap))
    ecg_data = []

    # create the samples for the ecg using the overlapping windows
    for i in range(0, len(ds_ecg) - int(window_size*(1-overlap)), int(window_size*(1-overlap))):
        ecg_data.append(ds_ecg[i:i+window_size])
    
    return ecg_data