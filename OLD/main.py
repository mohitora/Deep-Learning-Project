import yaml
import numpy as np
from models.model import build_model
from utils.load_data import prepare_data

def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def train_model():
    config = load_config()
    
    # Prepare data
    image_features, captions, tokenizer = prepare_data()
    
    # Build the model
    model = build_model(vocab_size=config['vocab_size'], 
                        embedding_dim=config['embedding_dim'], 
                        max_caption_length=config['max_caption_length'])
    
    # Train the model (you can split data into train and validation sets)
    model.fit([image_features, captions], captions, batch_size=config['batch_size'], epochs=config['epochs'])

if __name__ == "__main__":
    train_model()

