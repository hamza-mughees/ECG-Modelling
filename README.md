# Bayesian Modelling of the ECG

This project focuses on developing a Bayesian Neural Network model to analyze physiological signals, specifically Electrocardiogram (ECG) data. The primary goal is to accurately model ECG signals while quantifying the model's uncertainty. The implementation consists of a Convolutional Autoencoder architecture that processes ECG signals and applies approximate Bayesian inference using Monte Carlo dropout. This approach provides a robust way to model ECG signals, enabling better understanding and interpretation of the data while accounting for uncertainties. The project utilizes ECG data from the [EPHNOGRAM](https://physionet.org/content/ephnogram/1.0.0/) database, and the implemented model has undergone several iterations to optimize its performance.

## Prerequisites

- `pandas`
- `numpy`
- `matplotlib`
- `keras`
- `tensorflow`
- `sklearn`
- `scipy`

## Running the project

1. Clone the project by using `git clone https://github.com/hamza-mughees/ECG-Modelling.git`
2. Insall the data from the [EPHNOGRAM](https://physionet.org/content/ephnogram/1.0.0/) database and place it into the root directory.
3. Create a `res` directory in root.
4. Navigate into the `src` directory.
5. Create the data:
   1. For a single patient, run `python create_data_singlePatient.py`
   2. For all patients, run `python create_data_allPatients.py`
6. Train the model: run `python autoencoder.py`.
   1. For bayesian inference, make sure to assign `True` to the `bayes` variable in `globals.py`.
7. To analyze a trained model, run `performance.py`. Update the ID to that of the model in the `out` directory. The latest model would always be the one at the very bottom.