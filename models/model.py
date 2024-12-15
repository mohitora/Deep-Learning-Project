import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Add, Input
from tensorflow.keras.applications import InceptionV3

def build_model(vocab_size, embedding_dim, max_caption_length, image_size=224):
    # CNN for image feature extraction
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
    for layer in base_model.layers:
        layer.trainable = False
    image_input = base_model.input
    image_output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    
    # RNN for caption generation
    caption_input = Input(shape=(max_caption_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(caption_input)
    lstm = LSTM(256)(embedding)
    
    # Combine image features and caption features
    decoder_input = Add()([image_output, lstm])
    output = Dense(vocab_size, activation='softmax')(decoder_input)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

