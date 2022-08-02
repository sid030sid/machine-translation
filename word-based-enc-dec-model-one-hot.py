import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, RepeatVector, GRU, Dense, TimeDistributed, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# import corpuss
eng = pd.read_csv("data/europarl-v7.nl-en.en", sep="\t", names=["eng"])
nl = pd.read_csv("data/europarl-v7.nl-en.nl", sep="\t", names=["nl"], on_bad_lines='skip') #bad lines are skiped due to pandas.errors.ParserError: Expected 1 fields in line 952982, saw 2
corpus = pd.DataFrame(eng)
corpus["nl"] = nl

# select random 10% of corpus for tasks 2-4
## note: due to limited resources, 5% of the short sentences (<= 50 words) are taken as a sample
idx_short_sentences = [idx for idx, sentence in enumerate(corpus.nl) if len(sentence.split(" ")) <= 50] # the short sentence are determined by the dutch corpus as dutch tends to use more words than the english equivalent
corpus_short = corpus.filter(items=idx_short_sentences, axis=0)
sample = corpus_short.sample(round(len(corpus_short) * 0.05))

#PRE-PROCESSING (task 2)
# advised pre-processing step: remove lines with XML-Tags (starting with "<") --> there are no lines with xml tags if one manually filters the whole corpus for it
# lowercase the text
sample["eng"] = sample["eng"].str.lower() 
sample["nl"] = sample["nl"].str.lower()

# tokenization with keras
def tokenize(data): 
    tokenizer = Tokenizer() # set to Tokenizer(char_level = False) for char-based model???
    tokenizer.fit_on_texts(data)
    return tokenizer.texts_to_sequences(data), tokenizer

sample.eng, eng_tokenizer = tokenize(sample.eng)
sample.nl, nl_tokenizer = tokenize(sample.nl)

# padding sentences, making them all the same length
def pad(data, length=None):
    if length is None:
        length = max([len(sentence) for sentence in data])
    return pad_sequences(data, maxlen = length, padding = 'post')

sample.eng = [sent for sent in pad(sample.eng)]
sample.nl = [sent for sent in pad(sample.nl)]

# reversing inputted language which makes connection to encoder to decoder stronger
sample.eng = [sentence[::-1] for sentence in sample.eng]

# train and test set
indxs = np.arange(len(sample))
np.random.shuffle(indxs)
train_size = round(len(sample) * 0.8)
train_indxs = indxs[:train_size]
test_indxs = indxs[train_size:]

eng_train_set = [sample.iloc[idx].eng for idx in train_indxs]
eng_test_set = [sample.iloc[idx].eng for idx in test_indxs]
nl_train_set = [sample.iloc[idx].nl for idx in train_indxs]
nl_test_set = [sample.iloc[idx].nl for idx in test_indxs]




# RNN machine translator with one hot vectorization and character level https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
latent_dim = 10
num_encoder_tokens = 
num_decoder_tokens = 

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define and compile the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

# train and test model
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
batch_size = 65
epochs = 3
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)