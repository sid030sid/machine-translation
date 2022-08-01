# all tasks are done in this blog post: https://towardsdatascience.com/how-to-build-an-encoder-decoder-translation-model-using-lstm-with-python-and-keras-a31e9d864b9b
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, RepeatVector, GRU, Dense, TimeDistributed, Embedding
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import dataframe_image as dfi


# import corpuss
eng = pd.read_csv("data/europarl-v7.nl-en.en", sep="\t", names=["eng"])
nl = pd.read_csv("data/europarl-v7.nl-en.nl", sep="\t", names=["nl"], on_bad_lines='skip') #bad lines are skiped due to pandas.errors.ParserError: Expected 1 fields in line 952982, saw 2
corpus = pd.DataFrame(eng)
corpus["nl"] = nl

# TASK 1: analysis of corpus
## --> use code under vocubalary section of article: https://towardsdatascience.com/neural-machine-translation-with-python-c2f0a34f7dd
# number of words
words_eng = [word for sentence in corpus.eng for word in sentence.split()]
words_nl = [word for sentence in corpus.nl for word in sentence.split()]

# compare length of same sentence in english and in dutch
num_word_per_sentence_eng = [len(sentence.split(" ")) for sentence in corpus.eng]
num_word_per_sentence_nl = [len(sentence.split(" ")) for sentence in corpus.nl]

summary_corpus = pd.DataFrame({
    "Language":["English", "Dutch"], 
    "Number of sentences" : [len(corpus.eng), len(corpus.nl)],
    "Number of words" : [len(words_eng), len(words_nl)],
    "Minimum number of words per sentence" : [min(num_word_per_sentence_eng), min(num_word_per_sentence_nl)],
    "Average number of words per sentence" : [np.mean(num_word_per_sentence_eng), np.mean(num_word_per_sentence_nl)],
    "Maximum number of words per sentence" : [max(num_word_per_sentence_eng), max(num_word_per_sentence_nl)],
}) 
# eventually add: 
# "Number of words (no duplicates)" : [len(set(words_eng)), len(set(words_nl))],
# "Most frequent word" : [max(set(words_eng), key = words_eng.count), max(set(words_nl), key = words_nl.count)] --> takes too much cpu power

dfi.export(summary_corpus, "documentation/tables_as_image/summary_corpus.png")


# select random 10% of corpus for tasks 2-4
## note: due to limited resources, 5% of the short sentences (<= 50 words) are taken as a sample
idx_short_sentences = [idx for idx, sentence in enumerate(corpus.nl) if len(sentence.split(" ")) <= 50] # the short sentence are determined by the dutch corpus as dutch tends to use more words than the english equivalent
corpus_short = corpus.filter(items=idx_short_sentences, axis=0)
sample = corpus_short.sample(round(len(corpus_short) * 0.05))

# analysis of sample
eng_len = np.mean([len(sentence.split(" ")) for sentence in sample.eng]) # average amount of words in a sentence of englisch corpus
eng_vocab = len(set([word for sentence in sample.eng for word in sentence.split(" ")])) # number of words in english corpus (no duplicates!)
eng_num_word = len([word for sentence in sample.eng for word in sentence.split(" ")]) # numbe rof words in english corpus

nl_len = np.mean([len(sentence.split(" ")) for sentence in sample.nl]) # average amount of words in a sentence of dutch corpus
nl_vocab = len(set([word for sentence in sample.nl for word in sentence.split(" ")])) # number of words in dutch corpus (no duplicates!) 
nl_num_word = len([word for sentence in sample.nl for word in sentence.split(" ")]) # number of words in dutch corpus

summary_sample = pd.DataFrame({
    "Language":["English", "Dutch"], 
    "Number of sentences" : [len(sample.eng), len(sample.nl)],
    "Average number of words per sentence" : [eng_len, nl_len],
    "Number of words" : [eng_num_word, nl_num_word],
    "Number of words (no duplicates)" : [eng_vocab, nl_vocab]
})
dfi.export(summary_sample, "documentation/tables_as_image/summary_sample.png")

# TASK 2: pre-process sample
# --> follow: https://campus.datacamp.com/courses/machine-translation-in-python/training-and-generating-translations?ex=1
# advised pre-processing step: strip empty lines and their correspondences --> there are no empty lines
print("number of emtpy english sentences in sample", len(sample.eng[sample.eng == ""] != 0))
print("number of emtpy dutch sentences in sample", len(sample.eng[sample.nl == ""] != 0))    

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

#eng_max_sentence_length = max([len(sentence) for sentence in sample.eng])
#nl_max_sentence_length = max([len(sentence) for sentence in sample.nl])

sample.eng = [sent for sent in pad(sample.eng)]
sample.nl = [sent for sent in pad(sample.nl)]

# reversing inputted language which makes connection to encoder to decoder stronger
sample.eng = [sentence[::-1] for sentence in sample.eng]



# TASK 3: comparing machine translation models (RNN based vs. character based) depending on the word embeding (frequency based = Document-Term Matrix vs. prediction based = Word2Vec)
# create train and test set
indxs = np.arange(len(sample))
np.random.shuffle(indxs)
train_size = round(len(sample) * 0.8)
train_indxs = indxs[:train_size]
test_indxs = indxs[train_size:]

eng_train_set = [sample.iloc[idx].eng for idx in train_indxs]
eng_test_set = [sample.iloc[idx].eng for idx in test_indxs]
nl_train_set = [sample.iloc[idx].nl for idx in train_indxs]
nl_test_set = [sample.iloc[idx].nl for idx in test_indxs]

# word embeding
#embedding = Embedding(french_vocab_size, 64, input_length=input_shape[1]) 

    # embed corpus using one-hot vectors (frequency based)
    # embed corpus using Document-Term Matrix (frequency based)
    # embed corpus using continues bag of words (word2vec [prediciton based])
    # embed corpus using skip gram (word2vec [prediction based])

# RNN encoder-decoder model with one-hot vectorization
## input_lang and output_lang are matrices as list in list
def one_hot_encdec(input_lang, output_lang):
    num_hidden_units = 48

    # encoder
    encoder_inputs = Input(shape=(None, len(input_lang[1]))) # takes input of shape: length of sample.eng * eng_max_sentence_length
    encoder = GRU(num_hidden_units, return_state=True) # GRU model as layer which returns the state
    encoder_out, encoder_state = encoder(encoder_inputs) # Get the output and state from the GRU

    # decoder
    decoder_inputs = RepeatVector(len(output_lang[1]))(encoder_state) # RepeatVector layer: for every word of the output language the context vector/state is assigned
    decoder = GRU(num_hidden_units, return_sequences=True) # GRU model as layer that returns all outputs
    decoder_outputs = decoder(decoder_inputs, initial_state=encoder_state) # Get the outputs of the decoder
    decoder_dense_time = TimeDistributed(Dense(len(output_lang[1]), activation='softmax')) # Dense and TimeDistributed layers to get the final predictions (i.e. predicted dutch word probabilities) of the encoder-decoder model
    decoder_pred = decoder_dense_time(decoder_outputs) # Get the final prediction of the model

    # built model and compile it with with optimizer and loss function
    model = Model(inputs=encoder_inputs, outputs=decoder_pred)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

eng2dutch_one_hot_model = one_hot_encdec(eng_train_set, nl_train_set)

# train and test model
history = eng2dutch_one_hot_model.fit(
    np.array(eng_train_set),
    np.array(nl_train_set),
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(eng_test_set, nl_test_set),
)
results = eng2dutch_one_hot_model.evaluate(eng_test_set, nl_test_set, batch_size=128)

#old:
n_epochs, bsize = 3, 250
for ei in range(n_epochs):
  for i in range(0, train_size, bsize):
    # Get a single batch of inputs and outputs
    en_x = eng_train_set[i:i+bsize]   
    de_y = nl_train_set[i:i+bsize]

    # Train the model on a single batch of data
    model.train_on_batch(en_x, de_y)  

  # Evaluate the trained model with test set
  res = model.evaluate(eng_test_set, nl_test_set, verbose=0)
  print("{} => Loss:{}, Val Acc: {}".format(ei+1,res[0], res[1]*100.0))



# RNN encoder-decoder model with Keras embedding (Works like this https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
num_hidden_units = 48

en_inputs = Input(shape=(eng_len,)) # input layer which accepts a sequence of word IDs
en_emb = Embedding(eng_vocab, 96, input_length=eng_len)(en_inputs) # Embedding layer which accepts en_inputs
en_out, en_state = GRU(num_hidden_units, return_state=True)(en_emb)

de_inputs = RepeatVector(nl_len)(encoder_state)
de_out, _ = GRU(num_hidden_units, return_sequences=True, return_state=True)(de_inputs, initial_state=en_state) # GRU model as layer that returns all outputs
de_pred = TimeDistributed(Dense(nl_vocab, activation='softmax'))(de_out)

#built model and compile it with with optimizer and loss function
model_emb = Model(inputs=encoder_inputs, outputs=de_pred)
model_emb.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



# RNN encoder-decoder model with Glove embedding (https://keras.io/examples/nlp/pretrained_word_embeddings/)
# or: https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/



# English to Dutch neural machine translation 
    # RNN encoder-decoder model based on words
    # RNN encoder-decoder model based on characters

# Dutch to English  neural machine translation 
    # RNN encoder-decoder model based on words
    # RNN encoder-decoder model based on characters

# train and test 

# task 4: neural machine translation with attention
## maybe this a good source, depends if teacher force == attention based https://app.datacamp.com/learn/courses/machine-translation-in-python
## better source: https://www.tensorflow.org/text/tutorials/nmt_with_attention