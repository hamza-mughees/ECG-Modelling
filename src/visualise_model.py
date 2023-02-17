import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model_id = '20230217-154431'

model = load_model(f'../out/{model_id}/autoencoder.h5')
plot_model(model, to_file=f'../out/{model_id}/model_architecture.png', show_shapes=True, show_layer_names=True)
