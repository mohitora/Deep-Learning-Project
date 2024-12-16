import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .preprocess import preprocess_image  # Module to preprocess image files
from .tokenizer import create_tokenizer   # Module to create a tokenizer for text data
import yaml  # Library to handle YAML configuration files

def load_config():
    """
    Load configuration parameters from the YAML file.
    The configuration file contains paths and other essential settings for the model.
    
    Returns:
        dict: Configuration parameters loaded from 'config/config.yaml'.
    """
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_data():
    """
    Load images and their corresponding captions from the dataset directory.
    This function reads the dataset path from the configuration file and collects
    image file paths and captions.

    Returns:
        tuple: A pair containing:
            - images (list): List of image file paths.
            - captions (list): List of captions corresponding to the images.
    """
    config = load_config()  # Load the dataset path from the config file
    dataset_path = config['dataset_path']
    
    images = []   # To store image file paths
    captions = [] # To store captions for each image
    
    # Iterate through the dataset directory
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)  # Full path to the image file
        captions.append("A description of the image.")  # Placeholder captions (replace with real captions)
        images.append(img_path)  # Append the image path
        
    return images, captions

def prepare_data():
    """
    Prepares the data for training the image captioning model.
    It includes loading images and captions, tokenizing captions,
    padding sequences, and preprocessing image data.

    Steps:
        1. Load image paths and captions using `load_data`.
        2. Tokenize captions and convert them to numerical sequences.
        3. Pad the caption sequences to make them uniform in length.
        4. Preprocess image files to extract image features.

    Returns:
        tuple: A tuple containing:
            - image_features (numpy.ndarray): Preprocessed feature vectors of images.
            - sequences (numpy.ndarray): Padded numerical sequences of captions.
            - tokenizer (Tokenizer): Keras Tokenizer object for text processing.
    """
    # Step 1: Load image paths and captions
    images, captions = load_data()
    
    # Step 2: Create a tokenizer for the captions
    tokenizer = create_tokenizer(captions, vocab_size=5000)  # Limit vocabulary size to 5000 words
    
    # Step 3: Convert captions into sequences of integers
    sequences = tokenizer.texts_to_sequences(captions)  # Convert text to tokenized sequences
    max_len = max(len(seq) for seq in sequences)  # Determine the maximum sequence length
    sequences = pad_sequences(sequences, maxlen=max_len, padding='post')  # Pad sequences with zeros at the end
    
    # Step 4: Preprocess images to extract feature vectors
    # Each image is preprocessed into a feature vector suitable for model input
    image_features = [preprocess_image(img_path) for img_path in images]
    image_features = np.concatenate(image_features, axis=0)  # Combine all feature vectors into a single array
    
    return image_features, sequences, tokenizer
