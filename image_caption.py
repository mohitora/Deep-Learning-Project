from config import *

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

# --- Data Preparation ---

# Load the dataset containing image paths and corresponding captions
data = pd.read_csv(f"{DATA_PATH}/{IMAGE_CAPTIONS_FILE}") 

# Create an image data generator for data augmentation 
# (e.g., random rotations, flips, zooming) to increase data variability
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to the range [0, 1]
    preprocessing_function=preprocess_input,  # Apply InceptionV3's specific preprocessing
    shear_range=0.2,  # Randomly shear images by up to 20%
    zoom_range=0.2,   # Randomly zoom images by up to 20%
    horizontal_flip=True  # Randomly flip images horizontally
)

# Generate batches of images from the dataframe 
train_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    x_col='image_path', 
    y_col=None,  # We don't need labels for image generation
    target_size=IMAGE_SIZE,  # Use IMAGE_SIZE from config
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=True
)

# Preprocess Captions
# 1. Tokenize and build vocabulary
all_captions = []
for caption in data['caption']:
    all_captions.append("<start> " + caption + " <end>")  # Add start and end tokens

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_FEATURES, oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)  # Create a word-to-index mapping
vocab_size = len(tokenizer.word_index) + 1 

# 2. Create sequences of words for training 
input_sequences = []
for caption in all_captions:
    token_list = tokenizer.texts_to_sequences([caption])[0]  # Convert caption to a list of integers
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]  # Create input-output pairs (e.g., "start a" -> "a")
        input_sequences.append(n_gram_sequence)

# 3. Pad sequences to ensure uniform length 
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=MAX_LENGTH, padding='post'
)

# 4. Create features (input sequences) and labels (next word in the sequence)
features = padded_sequences[:, :-1]  # Input sequences without the last word
labels = padded_sequences[:, -1]     # Last word in each sequence

# --- Model Definition ---

# Create Embedding Layer to convert words into dense vectors
embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LENGTH-1)

# Define CNN Model (InceptionV3) for image feature extraction
image_model = InceptionV3(weights=INCEPTION_V3_WEIGHTS, include_top=False)  # Load pre-trained InceptionV3
image_input = Input(shape=IMAGE_SIZE + (3,))  # Use IMAGE_SIZE from config
encoded_image = image_model(image_input)  # Extract image features
encoded_image = tf.keras.layers.GlobalAveragePooling2D()(encoded_image)  # Pool features for better representation

# Define RNN Model (LSTM) for processing sequential text data
lstm_input = Input(shape=(MAX_LENGTH-1,))
embedded_sequence = embedding_layer(lstm_input) 
lstm_out, _ = LSTM(LSTM_UNITS, return_sequences=False)(embedded_sequence)  # Process the sequence

# Merge Image and Text Features
decoder_input = Concatenate()([encoded_image, lstm_out]) 

# Output Layer
outputs = Dense(vocab_size, activation='softmax')(decoder_input)  # Predict the probability distribution of the next word

# Create and Compile the Model
model = Model(inputs=[image_input, lstm_input], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# --- Model Training ---

for epoch in range(EPOCHS):
    for i in range(len(train_generator)):
        images, _ = train_generator.next()
        targets = labels[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
        model.train_on_batch([images, features[i * BATCH_SIZE: (i+1) * BATCH_SIZE]], targets)

# --- Caption Generation ---

def generate_caption(image_path):
    """
    Generates a caption for the given image path.

    Args:
        image_path: Path to the image file.

    Returns:
        Generated caption as a string.
    """
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    text_input = np.zeros((1, MAX_LENGTH-1))
    text_input[0, 0] = tokenizer.word_index['<start>']  # Initialize with the start token

    generated_caption = ''

    for i in range(MAX_LENGTH-1):
        prediction = model.predict([image, text_input], verbose=0)
        predicted_id = np.argmax(prediction)  # Get the index of the most probable word
        predicted_word = tokenizer.index_word[predicted_id]

        if predicted_word == '<end>':
            break

        generated_caption += predicted_word + ' '

        text_input[0, i+1] = predicted_id  # Update the input sequence with the predicted word

    return generated_caption

# Example Usage
generated_caption = generate_caption(IMAGE_PATH) 
print(generated_caption)