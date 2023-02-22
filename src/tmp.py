from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, Reshape, AveragePooling1D, Conv1DTranspose, UpSampling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import LogCosh, Huber, MeanSquaredError
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

model_settings = {
  'optimizer': Adam(learning_rate=0.001),
  'loss': 'mse',
  'encode_activations': [
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'relu'
  ],
  'decode_activations': [
    'relu',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU',
    'LeakyReLU'
  ],
}

# define the input layer and encoding dimensions of the autoencoder
encoding_dim = 32
input_layer = Input(shape=(224,))

# create the encoder layers of the autoencoder
encoded = Dense(512, activation=model_settings['encode_activations'][0])(input_layer)
encoded = Dense(256, activation=model_settings['encode_activations'][1])(encoded)
encoded = Reshape((256, 1))(encoded)
encoded = Conv1D(16, kernel_size=3, padding='same', activation=model_settings['encode_activations'][2])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Conv1D(16, kernel_size=3, padding='same', activation=model_settings['encode_activations'][3])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Conv1D(8, kernel_size=3, padding='same', activation=model_settings['encode_activations'][4])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Conv1D(4, kernel_size=3, padding='same', activation=model_settings['encode_activations'][5])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Conv1D(2, kernel_size=3, padding='same', activation=model_settings['encode_activations'][6])(encoded)
encoded = Flatten()(encoded)

# create the decoder layers of the autoencoder
decoded = Reshape((16, 2))(encoded)
decoded = Conv1DTranspose(4, kernel_size=3, padding='same', activation=model_settings['decode_activations'][0])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(8, kernel_size=3, padding='same', activation=model_settings['decode_activations'][1])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(16, kernel_size=3, padding='same', activation=model_settings['decode_activations'][2])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(16, kernel_size=3, padding='same', activation=model_settings['decode_activations'][3])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(1, kernel_size=3, padding='same', activation=model_settings['decode_activations'][4])(decoded)
decoded = Flatten()(decoded)
decoded = Dense(512, activation=model_settings['decode_activations'][5])(decoded)

# define the output layer of the autoencoder
output_layer = Dense(224, activation=model_settings['decode_activations'][5])(decoded)

# create the autoencoder model
autoencoder = Model(input_layer, output_layer)

# compile the model
autoencoder.compile(optimizer=model_settings['optimizer'], loss=model_settings['loss'])

# print the model summary
print(autoencoder.summary())