import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from keras.layers import Input, RepeatVector, GRU, Dense, TimeDistributed, Embedding, LSTM, Activation
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# import corpuss
eng = pd.read_csv("data/europarl-v7.nl-en.en", sep="\t", names=["eng"])
nl = pd.read_csv("data/europarl-v7.nl-en.nl", sep="\t", names=["nl"], on_bad_lines='skip') #bad lines are skiped due to pandas.errors.ParserError: Expected 1 fields in line 952982, saw 2
corpus = pd.DataFrame(eng)
corpus["nl"] = nl

# select random 10% of corpus for tasks 2-4
## note: due to limited resources, 5% of the short sentences (<= 50 words) are taken as a sample
idx_short_sentences = [idx for idx, sentence in enumerate(corpus.nl) if len(sentence.split(" ")) <= 20] # the short sentence are determined by the dutch corpus as dutch tends to use more words than the english equivalent
corpus_short = corpus.filter(items=idx_short_sentences, axis=0)
sample = corpus_short.sample(round(len(corpus_short) * 0.03))

eng_sentences = np.array(sample.eng)
nl_sentences = np.array(sample.nl)

def clean_sentence(sentence):
    # Lower case the sentence
    lower_case_sent = sentence.lower()
    # Strip punctuation
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string.punctuation))
    return clean_sentence

def tokenize(sentences):
    # Create tokenizer
    text_tokenizer = Tokenizer()
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

# Clean sentences
eng_sentences_clean = [clean_sentence(sentence) for sentence in eng_sentences]
nl_sentences_clean = [clean_sentence(sentence) for sentence in nl_sentences]

# Tokenize words
nl_text_tokenized, nl_text_tokenizer = tokenize(nl_sentences_clean)
eng_text_tokenized, eng_text_tokenizer = tokenize(eng_sentences_clean)

# Analysis of sample
print('Maximum length dutch sentence: {}'.format(len(max(nl_text_tokenized,key=len))))
print('Maximum length english sentence: {}'.format(len(max(eng_text_tokenized,key=len))))


# Check language length
nl_vocab = len(nl_text_tokenizer.word_index) + 1
eng_vocab = len(eng_text_tokenizer.word_index) + 1
print("Dutch vocabulary is of {} unique words".format(nl_vocab))
print("English vocabulary is of {} unique words".format(eng_vocab))

# Respective max sentence length
max_nl_len = int(len(max(nl_text_tokenized,key=len)))
max_eng_len = int(len(max(eng_text_tokenized,key=len)))

# padding to respective max sentence senght
nl_pad_sentence = pad_sequences(nl_text_tokenized, max_nl_len, padding = "post")
eng_pad_sentence = pad_sequences(eng_text_tokenized, max_eng_len, padding = "post")

nl_pad_sentence = nl_pad_sentence.reshape(*nl_pad_sentence.shape, 1)

# model
def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

    model = Sequential()
    model.add(GRU(64, input_shape = input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

tmp_x = eng_pad_sentence.reshape((-1, eng_pad_sentence.shape[1], 1))

encodeco_model = encdec_model(
    tmp_x.shape,
    nl_pad_sentence.shape[1],
    eng_vocab,
    nl_vocab)
max_nl_len

encodeco_model.summary()

history = encodeco_model.fit(tmp_x, nl_pad_sentence, batch_size=32, epochs=3, validation_split=0.2)
print(history.history['accuracy'])
# visualisation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def approach1():
    # Reshape data to fit input layer of model
    nl_pad_sentence = nl_pad_sentence.reshape(*nl_pad_sentence.shape, 1)
    eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

    # one hot encode
    #from keras.utils import to_categorical
    #eng_one_hot = to_categorical(eng_pad_sentence , eng_vocab)
    #nl_one_hot = to_categorical(nl_pad_sentence , nl_vocab)

    # MODEL
    # encoder
    input_sequence = Input(shape=(max_eng_len,))
    embedding = Embedding(input_dim=eng_vocab, output_dim=128,)(input_sequence)
    encoder = LSTM(64, return_sequences=False)(embedding)

    # decoder
    r_vec = RepeatVector(max_nl_len)(encoder) # encoder's size should e batch_size X input_size
    decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
    logits = TimeDistributed(Dense(nl_vocab))(decoder)

    # build and compile model
    enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
    enc_dec_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    enc_dec_model.summary()

    # train and test
    model_results = enc_dec_model.fit(eng_pad_sentence, nl_pad_sentence, batch_size=30, epochs=100)
    nl_pad_sentence.shape
    max_nl_len

    eng_pad_sentence.shape
    max_eng_len