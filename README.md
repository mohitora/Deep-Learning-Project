# Image Captioning with Deep Learning

This project implements an image captioning model using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The 1  model takes an image as input and generates a textual description (caption) of its content

## Dependencies
This project requires the following Python libraries:

- tensorflow
- numpy
- pandas
- nltk

## Usage
### 1. Prepare Data:

Create a CSV file named image_captions.csv with two columns:

image_path: Path to the image file (relative or absolute)
caption: Textual description of the image
Place your images in a dedicated folder (e.g., images).

Update the image_path column in image_captions.csv to reflect the actual image locations.

### 2. Run the Script:

Open a terminal and navigate to the project directory.
Run the following command:
``` bash
python image_captioning.py
```

### 3. Generate Captions:

The script will train the model based on the provided data.
After training, the script demonstrates caption generation for a sample image path (specified in config.py).
You can modify the IMAGE_PATH variable in config.py to generate captions for other images in your dataset.
Configuration

The script utilizes a separate configuration file (config.py) to manage key parameters. You can customize the following settings in config.py:
```
DATA_PATH: Path to the directory containing image_captions.csv.
IMAGE_CAPTIONS_FILE: Name of the CSV file containing image-caption pairs.
IMAGE_PATH: Path to the sample image for caption generation (modify this as needed).
IMAGE_SIZE: Target size for resizing images (e.g., (299, 299))
BATCH_SIZE: Number of images processed simultaneously during training.
MAX_LENGTH: Maximum length of captions (controls padding).
MAX_FEATURES: Maximum number of words considered in the vocabulary.
EMBEDDING_DIM: Dimensionality of word embeddings.
EPOCHS: Number of training epochs.
INCEPTION_V3_WEIGHTS: Weights used for pre-trained InceptionV3 model.
LSTM_UNITS: Number of units in the LSTM layer.
```
### 
4. Additional Notes

This script provides a basic implementation of image captioning.
You can experiment with different hyperparameters, architectures, and data augmentation techniques to improve performance.
Consider exploring techniques like beam search for more diverse caption generation.


### References
https://github.com/karpathy/neuraltalk2
https://arxiv.org/abs/1411.4555