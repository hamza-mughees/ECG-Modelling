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
encoded = Reshape((224, 1))(input_layer)
encoded = Conv1D(2, kernel_size=3, padding='same', activation=model_settings['encode_activations'][0])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Conv1D(4, kernel_size=3, padding='same', activation=model_settings['encode_activations'][1])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Conv1D(8, kernel_size=3, padding='same', activation=model_settings['encode_activations'][2])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Dropout(rate=0.2)(encoded)
encoded = Conv1D(16, kernel_size=3, padding='same', activation=model_settings['encode_activations'][3])(encoded)
encoded = AveragePooling1D(pool_size=2)(encoded)
encoded = Conv1D(32, kernel_size=3, padding='same', activation=model_settings['encode_activations'][8])(encoded)
encoded = Flatten()(encoded)
encoded = Dense(512, activation=model_settings['encode_activations'][5])(encoded)
encoded = Dense(256, activation=model_settings['encode_activations'][6])(encoded)
encoded = Dense(128, activation=model_settings['encode_activations'][7])(encoded)
encoded = Dense(64, activation=model_settings['encode_activations'][9])(encoded)
encoded = Dense(encoding_dim, activation=model_settings['encode_activations'][9])(encoded)

# create the decoder layers of the autoencoder
decoded = Dense(64, activation=model_settings['decode_activations'][0])(encoded)
decoded = Dense(128, activation=model_settings['decode_activations'][1])(decoded)
decoded = Dense(256, activation=model_settings['decode_activations'][2])(decoded)
decoded = Dense(512, activation=model_settings['decode_activations'][3])(decoded)
decoded = Dense(448, activation=model_settings['decode_activations'][4])(decoded)
decoded = Reshape((14, 32))(decoded)
decoded = Conv1DTranspose(16, kernel_size=3, padding='same', activation=model_settings['decode_activations'][5])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(8, kernel_size=3, padding='same', activation=model_settings['decode_activations'][6])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(4, kernel_size=3, padding='same', activation=model_settings['decode_activations'][7])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(2, kernel_size=3, padding='same', activation=model_settings['decode_activations'][8])(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1DTranspose(1, kernel_size=3, padding='same', activation=model_settings['decode_activations'][9])(decoded)

# define the output layer of the autoencoder
output_layer = Flatten()(decoded)

# create the autoencoder model
autoencoder = Model(input_layer, output_layer)

# compile the model
autoencoder.compile(optimizer=model_settings['optimizer'], loss=model_settings['loss'])

# print the model summary
print(autoencoder.summary())