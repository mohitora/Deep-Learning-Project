import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

