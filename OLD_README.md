# Image Captioning using CNN

This project demonstrates how to use a Convolutional Neural Network (CNN) combined with a Recurrent Neural Network (RNN) for generating captions for images. The CNN model extracts features from images, while the RNN (specifically an LSTM) is used to generate captions based on the image features.

The project uses TensorFlow and Keras for building the model, and the configuration is managed through a YAML file.

## Requirements

The following libraries are required to run the project:

- TensorFlow (version 2.8.0)
- Keras (version 2.8.0)
- PyYAML
- NumPy
- Matplotlib
- Pillow

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```
## Project Structure
```Image_Captioning/
│
├── config/
│   └── config.yaml            # Configuration file with project parameters
├── data/
│   └── <your dataset files>   # Dataset containing images and captions
├── models/
│   └── model.py               # CNN-RNN model definition
├── utils/
│   └── preprocess.py          # Image preprocessing utilities
│   └── tokenizer.py           # Tokenizer for captions
│   └── load_data.py           # Data loading and preparation
├── main.py                    # Main script to run the model training
└── requirements.txt           # Required libraries
```

## Configuration

The project uses a config.yaml file to store configuration parameters, including:

- dataset_path: The path to the dataset directory containing images and captions.
- batch_size: The batch size for training.
- epochs: The number of epochs to train the model.
- learning_rate: The learning rate for the optimizer.
- image_size: The target size to which images will be resized.
- embedding_dim: The dimension of the word embeddings.
- vocab_size: The vocabulary size for the tokenizer.
- max_caption_length: The maximum length of captions.

### Example of config.yaml
```
dataset_path: "/path/to/dataset"
batch_size: 32
epochs: 20
learning_rate: 0.001
image_size: 224
embedding_dim: 256
vocab_size: 5000
max_caption_length: 50
```
## Dataset
The dataset should consist of images paired with captions. The dataset should be structured in a way where each image has a corresponding caption. Ensure the images are placed in the ```data/ directory```, and modify the ```load_data.py``` script to handle the dataset properly.

## Training the Model
Once the dataset is ready, you can start training the model by running the main.py script:
```
bash
python main.py
```
This will load the dataset, preprocess the images and captions, build the model, and start the training process.

## Model Architecture
The model architecture consists of two main components:

- CNN for Image Feature Extraction: We use the InceptionV3 model pre-trained on ImageNet for extracting features from the images. These features are passed to the RNN.

- RNN for Caption Generation: The RNN is implemented using an LSTM layer. The LSTM is trained to predict the next word in the sequence based on the features extracted from the image and the previously generated words.

## Testing the Model
Once the model is trained, you can use it to generate captions for new images by feeding an image into the model and using the decoder to predict the caption.



