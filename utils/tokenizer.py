from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer(captions, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

