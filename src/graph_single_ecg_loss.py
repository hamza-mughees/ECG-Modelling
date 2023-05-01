# import numpy as np
import pandas as pd

from utils import loss_single_ecg

# load the data from the csv file into a pandas dataframe
df = pd.read_csv("../res/singlePatient.csv", header=None)

# convert the dataframe to a numpy array
ecg_data = df.to_numpy()

loss_single_ecg('20230224-131313', ecg_data)