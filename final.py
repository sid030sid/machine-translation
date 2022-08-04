import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, RepeatVector, GRU, Dense, TimeDistributed, Embedding, LSTM, Activation
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.initializers import Constant # for matrix with pre-trained embedding weights
from gensim.models import KeyedVectors # to load in pretrained word2vec weights
import dataframe_image as dfi

# import corpuss
eng = pd.read_csv("data/europarl-v7.nl-en.en", sep="\t", names=["eng"])
nl = pd.read_csv("data/europarl-v7.nl-en.nl", sep="\t", names=["nl"], on_bad_lines='skip') #bad lines are skiped due to pandas.errors.ParserError: Expected 1 fields in line 952982, saw 2
corpus = pd.DataFrame(eng)
corpus["nl"] = nl

# TASK 1: analysis of corpus
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
dfi.export(summary_corpus, "documentation/tables_as_image/summary_corpus.png")

# select random data of corpus for tasks 2-4
## note: due to limited resources, 3% of the short sentences (<= 50 words) are taken as a sample
nl_idx_short_sentences = [idx for idx, sentence in enumerate(corpus.nl) if len(sentence.split(" ")) <= 20] # the short sentence are determined by the dutch corpus as dutch tends to use more words than the english equivalent
eng_idx_short_sentences = [idx for idx, sentence in enumerate(corpus.eng) if len(sentence.split(" ")) <= 20]
idx_short_sentences = list(set(nl_idx_short_sentences) & set(eng_idx_short_sentences))
corpus_short = corpus.filter(items=idx_short_sentences, axis=0)
sample = corpus_short.sample(round(len(corpus_short) * 0.03))

# transform english and dutch corpus into single numpy arrays for frther steps
eng_sentences = np.array(sample.eng)
nl_sentences = np.array(sample.nl)

# TASK 2: pre-pprocessing
def clean_sentence(sentence):
    # Lower case the sentence
    lower_case_sent = sentence.lower()

    # Strip punctuation
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string.punctuation))
    return clean_sentence

def tokenize(sentences, character_based = False):
    # Create tokenizer
    text_tokenizer = Tokenizer(char_level=character_based)
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

# clean sentences
eng_sentences_clean = [clean_sentence(sentence) for sentence in eng_sentences]
nl_sentences_clean = [clean_sentence(sentence) for sentence in nl_sentences]

# word-based tokenization
nl_text_tokenized_word, nl_text_tokenizer_word = tokenize(nl_sentences_clean)
eng_text_tokenized_word, eng_text_tokenizer_word = tokenize(eng_sentences_clean)

# character-based tokenization
nl_text_tokenized_char, nl_text_tokenizer_char = tokenize(nl_sentences_clean, character_based=True)
eng_text_tokenized_char, eng_text_tokenizer_char = tokenize(eng_sentences_clean, character_based=True)

# respective max sentence length in words or characters: crucial for padding and building all upcoming models
max_nl_len_word = int(len(max(nl_text_tokenized_word,key=len)))
max_eng_len_word = int(len(max(eng_text_tokenized_word,key=len)))
max_nl_len_char = int(len(max(nl_text_tokenized_char,key=len)))
max_eng_len_char = int(len(max(eng_text_tokenized_char,key=len)))

# padding to respective max sentence length
nl_pad_sentence_word = pad_sequences(nl_text_tokenized_word, max_nl_len_word, padding = "post")
eng_pad_sentence_word = pad_sequences(eng_text_tokenized_word, max_eng_len_word, padding = "post")
nl_pad_sentence_char = pad_sequences(nl_text_tokenized_char, max_nl_len_char, padding = "post")
eng_pad_sentence_char = pad_sequences(eng_text_tokenized_char, max_eng_len_char, padding = "post")

# reshape to 3D due to Keras' requirements
nl_pad_sentence_word = nl_pad_sentence_word.reshape(*nl_pad_sentence_word.shape, 1)
eng_pad_sentence_word = eng_pad_sentence_word.reshape(*eng_pad_sentence_word.shape, 1)
nl_pad_sentence_char = nl_pad_sentence_char.reshape(*nl_pad_sentence_char.shape, 1)
eng_pad_sentence_char = eng_pad_sentence_char.reshape(*eng_pad_sentence_char.shape, 1)

# amount of unique words: crucial for building all upcoming models
nl_vocab = len(nl_text_tokenizer_word.word_index) + 1
eng_vocab = len(eng_text_tokenizer_word.word_index) + 1
nl_char = len(eng_text_tokenizer_char.word_index) + 1
eng_char = len(eng_text_tokenizer_char.word_index) + 1

# summary of sample for documentation purposes
eng_avg_len = np.mean([len(sentence) for sentence in eng_text_tokenized_word]) # average amount of words in a sentence of englisch corpus
eng_num_word = sum([len(sentence) for sentence in eng_text_tokenized_word]) # number of words in english corpus
eng_max_len = len(max(eng_text_tokenized_word,key=len))
eng_min_len = len(min(eng_text_tokenized_word,key=len))

nl_avg_len = np.mean([len(sentence) for sentence in nl_text_tokenized_word]) # average amount of words in a sentence of dutch corpus
nl_num_word = len([len(sentence) for sentence in eng_text_tokenized_word]) # number of words in dutch corpus
nl_max_len = len(max(nl_text_tokenized_word,key=len))
nl_min_len = len(min(nl_text_tokenized_word,key=len))

summary_sample = pd.DataFrame({
    "Language":["English", "Dutch"], 
    "Number of sentences" : [len(sample.eng), len(sample.nl)],
    "Minimum number of words per sentence" : [eng_min_len, nl_min_len],
    "Average number of words per sentence" : [eng_avg_len, nl_avg_len],
    "Maximum number of words per sentence" : [eng_max_len, nl_max_len],
    "Number of words" : [eng_num_word, nl_num_word],
    "Number of words (no duplicates)" : [eng_vocab, nl_vocab], 
    "Nmber of used characters (no duplicates)" : [eng_char, nl_char],
})
dfi.export(summary_sample, "documentation/tables_as_image/summary_sample.png")



# TASK 3: comparison of performance of neural machine translator depending word embedding and tokenization (word or character)
# no embedding:
# function for building ecoder-decoder model without embedding (only one-hot representation)
def encdec_model(input_shape, output_max_len, output_vocab_size):
    model = Sequential()
    model.add(GRU(64, input_shape = input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build word-based english to dutch and dutch to english translator
eng2nl_encdec_word_model = encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab)
nl2eng_encdec_word_model = encdec_model(nl_pad_sentence_word.shape, max_eng_len_word, eng_vocab)

# build character-based english to dutch and dutch to english translator
eng2nl_encdec_char_model = encdec_model(eng_pad_sentence_char.shape, max_nl_len_char, nl_char)
nl2eng_encdec_char_model = encdec_model(nl_pad_sentence_char.shape, max_eng_len_char, eng_char)



# keras' native embedding model:
# function for building ecoder-decoder model with Keras' embedding model
def keras_embd_encdec_model(input_shape, output_max_len, output_vocab_size, input_vocab_size):
    model = Sequential()
    model.add(Embedding(input_vocab_size, 32, input_length=input_shape[1], trainable = True))
    model.add(GRU(64, return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build word-based english to dutch and dutch to english translator with keras embedding
eng2nl_keras_embd_encdec_word_model = keras_embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab, eng_vocab)
nl2eng_keras_embd_encdec_word_model = keras_embd_encdec_model(nl_pad_sentence_word.shape, max_eng_len_word, eng_vocab, nl_vocab)

# build character-based english to dutch and dutch to english translator with keras embedding
eng2nl_keras_embd_encdec_char_model = keras_embd_encdec_model(eng_pad_sentence_char.shape, max_nl_len_char, nl_char, eng_char)
nl2eng_keras_embd_encdec_char_model = keras_embd_encdec_model(nl_pad_sentence_char.shape, max_eng_len_char, eng_char, nl_char)



# glove embedding:
# get all of glove's pre-trained weights
glove_word_vectors = {}
with open("data\glove.6B.300d.txt", encoding="utf8") as glove_data:
    for element in glove_data:
        values = element.split()            
        word = values[0]            
        coefs = values[1:]            
        glove_word_vectors[word] = np.asarray(coefs, dtype='float32')

# select glove vectors whose words are also present in the english corpus and store it in a matrix
glove_embedding_matrix = np.zeros((eng_vocab, 300)) # 300 = word vector dimension = number of weights per word in glove --> see: len(glove_word_vectors["the"])
words_not_in_glove = []
for word, i in eng_text_tokenizer_word.word_index.items():        
    embedding_vector = glove_word_vectors.get(word)
    if embedding_vector is not None: # non exisiting words will be zero            
        glove_embedding_matrix[i] = embedding_vector
    else : 
        words_not_in_glove.append(word)

# function for building ecoder-decoder model with Glove embedding
def glove_embd_encdec_model(input_shape, output_max_len, output_vocab_size, input_vocab_size):
    #build model
    model = Sequential()
    model.add(Embedding(input_vocab_size, 300, input_length=input_shape[1], embeddings_initializer=Constant(glove_embedding_matrix)))
    model.add(GRU(64, return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build models (only those capable of using pretrained glove: english as input language and word-based)
eng2nl_glove_embd_encdec_word_model = glove_embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab, eng_vocab)



# word2vec embedding:
# word2vec embedding based on pre-trained weights (source: https://code.google.com/archive/p/word2vec/):
word2vec = KeyedVectors.load_word2vec_format("data\GoogleNews-vectors-negative300.bin", binary=True) # get all pretrained weigths

# select word2vec vectors whose words are also present in the english corpus and store it in a matrix
word2vec_embedding_matrix = np.zeros((eng_vocab, 300)) # 300 = word2vec vector dimension (= 300 weigths per word)
words_not_in_word2vec = []
for word, i in eng_text_tokenizer_word.word_index.items():  
    if word in word2vec : 
        word2vec_embedding_matrix[i] = word2vec[word]
    else : 
        words_not_in_word2vec.append(word)

# function for building ecoder-decoder model with word2vec embedding
def word2vec_embd_encdec_model(input_shape, output_max_len, output_vocab_size, input_vocab_size):
    #build model
    model = Sequential()
    model.add(Embedding(input_vocab_size, 300, input_length=input_shape[1], embeddings_initializer=Constant(word2vec_embedding_matrix))) # 300 due to word2vec vector dimension
    model.add(GRU(64, return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build models (only those capable of using pretrained word2vec: english as input language and word-based)
eng2nl_word2vec_embd_encdec_word_model = word2vec_embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab, eng_vocab)



# TASK 4: neural machine translation with attention (based on guide: https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39)
#https://github.com/thushv89/attention_keras

#https://pypi.org/project/keras-self-attention/

#from attention_keras.src.layers.attention import AttentionLayer

# try to use kears' built in attention layer: https://www.google.com/search?q=how+to+use+keras%27+attention+layer&rlz=1C1CHBF_deDE912DE912&oq=how+to+use+keras%27+attention+layer&aqs=chrome..69i57j0i22i30j0i390l5.768642j0j9&sourceid=chrome&ie=UTF-8
def attention_encdec_model():
    model = Sequential()
    return model

# build models with attention mechanism


model_training_manuals = [
    {   
        "title" : "English to Dutch translator (word-based, Word2Vec embedding)",
        "model" : eng2nl_word2vec_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    },
    {   
        "title" : "English to Dutch translator (word-based, Glove embedding)",
        "model" : eng2nl_glove_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    }, 
    {   
        "title" : "English to Dutch translator (word-based)",
        "model" : eng2nl_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    },

    {
        "title" : "Dutch to English translator (word-based)",
        "model" : nl2eng_encdec_word_model,
        "X" : nl_pad_sentence_word,
        "y" : eng_pad_sentence_word
    },

    {
        "title" : "English to Dutch translator (char-based)", 
        "model" : eng2nl_encdec_char_model,
        "X" : eng_pad_sentence_char,
        "y" : nl_pad_sentence_char
    },

    {
        "title" :"Dutch to English translator (char-based)", 
        "model" : nl2eng_encdec_char_model,
        "X" : nl_pad_sentence_char,
        "y" : eng_pad_sentence_char
    }, 

    {
        "title" : "Dutch to English translator (word-based, Keras' embedding)",
        "model" : nl2eng_keras_embd_encdec_word_model,
        "X" : nl_pad_sentence_word,
        "y" : eng_pad_sentence_word
    },

    {   
        "title" : "English to Dutch translator (word-based, Keras' embedding)",
        "model" : eng2nl_keras_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    },
    {
        "title" : "English to Dutch translator (char-based, Keras' embedding)", 
        "model" : eng2nl_keras_embd_encdec_char_model,
        "X" : eng_pad_sentence_char,
        "y" : nl_pad_sentence_char
    },

    {
        "title" :"Dutch to English translator (char-based, Keras' embedding)", 
        "model" : nl2eng_keras_embd_encdec_char_model,
        "X" : nl_pad_sentence_char,
        "y" : eng_pad_sentence_char
    }
]


for manual in model_training_manuals:
    print("\t\t\t")
    print("MODEL: "+manual["title"])
    
    # show model architecture
    manual["model"].summary()

    # train and test model
    history = manual["model"].fit(manual["X"], manual["y"], batch_size=64, epochs=20, validation_split=0.2)

    # accuracy of validation of last epoch
    print("Last measured accuracy score: ", history.history["val_accuracy"][2])

    # visualise accuracy history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy of '+manual["title"])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("documentation/plots/accuracy/"+manual["title"]+".png", format="png")

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("documentation/plots/loss/"+manual["title"]+".png", format="png")