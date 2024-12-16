# Import the Tokenizer class from Keras' text preprocessing module
from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer(captions, vocab_size):
    # Create a Tokenizer object that will process text
    # num_words: The maximum number of words to keep in the tokenizer
    # oov_token: This token will be used for any out-of-vocabulary words
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    
    # Fit the tokenizer on the provided captions (this learns the word frequency)
    # It creates a word index, which maps each word to a unique integer
    tokenizer.fit_on_texts(captions)
    
    # Return the fitted tokenizer, which can now be used to process text
    return tokenizer