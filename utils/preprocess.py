import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocesses an image to prepare it for input into a pre-trained deep learning model.
    The function performs the following steps:
        1. Loads the image from the specified path and resizes it to the target dimensions.
        2. Converts the image into a NumPy array.
        3. Applies preprocessing specific to the InceptionV3 model (e.g., scaling and normalization).
        4. Expands the dimensions of the image array to add a batch size of 1.

    Args:
        img_path (str): Path to the image file to be processed.
        target_size (tuple): Target dimensions `(height, width)` to resize the image.
                            Default is `(224, 224)`.

    Returns:
        numpy.ndarray: A 4D array of shape `(1, target_size[0], target_size[1], 3)` ready for model input.
                       The array represents a single batch with one preprocessed image.
    """
    # Step 1: Load the image from the given path and resize it to the target size
    img = image.load_img(img_path, target_size=target_size)
    
    # Step 2: Convert the image to a NumPy array with pixel intensity values
    img_array = image.img_to_array(img)
    
    # Step 3: Apply InceptionV3-specific preprocessing (e.g., scaling and normalization)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    
    # Step 4: Expand dimensions to add a batch size of 1 (shape becomes 4D)
    return np.expand_dims(img_array, axis=0)


