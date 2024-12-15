import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .preprocess import preprocess_image
from .tokenizer import create_tokenizer
import yaml

def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_data():
    config = load_config()
    dataset_path = config['dataset_path']
    
    images = []
    captions = []
    
    # Loading image paths and captions (replace with your dataset)
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        captions.append("A description of the image.")  # Replace with real captions
        images.append(img_path)
        
    return images, captions

def prepare_data():
    images, captions = load_data()
    tokenizer = create_tokenizer(captions, vocab_size=5000)
    
    # Convert captions to sequences
    sequences = tokenizer.texts_to_sequences(captions)
    max_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    # Preprocess images
    image_features = [preprocess_image(img_path) for img_path in images]
    image_features = np.concatenate(image_features, axis=0)
    
    return image_features, sequences, tokenizer

