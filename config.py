# config.py

# Data paths
DATA_PATH = "/Users/mohit/Documents/14-PHD/github/Deep-Learning-Project/training_data"
IMAGE_CAPTIONS_FILE = "image_captions.csv"
IMAGE_PATH = "/Users/mohit/Documents/14-PHD/github/Deep-Learning-Project/images" 

# Image preprocessing
IMAGE_SIZE = (299, 299)  # InceptionV3 input size
BATCH_SIZE = 32

# Caption preprocessing
MAX_LENGTH = 50  # Maximum length of a caption
MAX_FEATURES = 10000  # Maximum number of words in the vocabulary
EMBEDDING_DIM = 256

# Model training
EPOCHS = 10

# InceptionV3 model settings
INCEPTION_V3_WEIGHTS = 'imagenet' 

# LSTM layer size
LSTM_UNITS = 256
