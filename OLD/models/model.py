import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to build the CNN encoder model
def build_cnn_encoder(image_size):
    """
    Builds a CNN model based on InceptionV3 to extract features from images.

    Args:
        image_size (int): Size to which the input images will be resized.

    Returns:
        tf.keras.Model: A CNN model that outputs image feature vectors.
    """
    # Load the pre-trained InceptionV3 model without the top classification layer
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

    # Freeze the base model to prevent updating its weights during training
    base_model.trainable = False

    # Add a global average pooling layer to reduce the spatial dimensions
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    # Define the model
    model = Model(inputs=base_model.input, outputs=x, name="CNN_Encoder")
    return model

# Function to build the RNN decoder model
def build_rnn_decoder(vocab_size, embedding_dim, max_caption_length):
    """
    Builds an RNN decoder using an embedding layer and LSTM for generating captions.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the word embeddings.
        max_caption_length (int): Maximum length of the captions.

    Returns:
        tf.keras.Model: An RNN model for caption generation.
    """
    # Input for the sequences of word indices
    input_seq = Input(shape=(max_caption_length,), name="Input_Sequence")

    # Embedding layer to convert word indices into dense vectors of fixed size
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="Embedding_Layer")(input_seq)

    # LSTM layer for sequential processing of captions
    x = LSTM(256, return_sequences=True, name="LSTM_Layer_1")(x)
    x = LSTM(256, return_sequences=False, name="LSTM_Layer_2")(x)

    # Fully connected layer to predict the next word
    output = Dense(vocab_size, activation="softmax", name="Dense_Output")(x)

    # Define the model
    model = Model(inputs=input_seq, outputs=output, name="RNN_Decoder")
    return model

# Function to build the full image captioning model
def build_model (image_size, vocab_size, embedding_dim, max_caption_length):
    """
    Combines the CNN encoder and RNN decoder to create the image captioning model.

    Args:
        image_size (int): Size to which the input images will be resized.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the word embeddings.
        max_caption_length (int): Maximum length of the captions.

    Returns:
        tf.keras.Model: The complete image captioning model.
    """
    # Build the CNN encoder
    cnn_encoder = build_cnn_encoder(image_size)

    # Build the RNN decoder
    rnn_decoder = build_rnn_decoder(vocab_size, embedding_dim, max_caption_length)

    # Inputs: image and caption sequence
    image_input = cnn_encoder.input  # Input to the CNN encoder
    caption_input = rnn_decoder.input  # Input to the RNN decoder

    # Extract image features using the CNN encoder
    image_features = cnn_encoder.output

    # Expand dimensions to match the caption sequence input shape
    image_features = tf.keras.layers.RepeatVector(max_caption_length, name="Repeat_Image_Features")(image_features)

    # Concatenate image features with word embeddings from the RNN
    combined_features = tf.keras.layers.Concatenate(name="Concatenate_Features")([image_features, rnn_decoder.input])

    # LSTM layer processes the combined features
    x = LSTM(256, return_sequences=False, name="LSTM_Combined_Features")(combined_features)

    # Fully connected layer to predict the next word
    output = Dense(vocab_size, activation="softmax", name="Final_Output")(x)

    # Define the complete model
    model = Model(inputs=[image_input, caption_input], outputs=output, name="Image_Captioning_Model")
    return model
