import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, RepeatVector, GRU, Dense, TimeDistributed, Embedding, LSTM, Activation
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.initializers import Constant

# import corpuss
eng = pd.read_csv("data/europarl-v7.nl-en.en", sep="\t", names=["eng"])
nl = pd.read_csv("data/europarl-v7.nl-en.nl", sep="\t", names=["nl"], on_bad_lines='skip') #bad lines are skiped due to pandas.errors.ParserError: Expected 1 fields in line 952982, saw 2
corpus = pd.DataFrame(eng)
corpus["nl"] = nl

# select random data of corpus for tasks 2-4
## note: due to limited resources, 3% of the short sentences (<= 20 words) are taken as a sample
#idx_short_sentences = [idx for idx, sentence in enumerate(corpus.nl) if len(sentence.split(" ")) <= 20] # the short sentence are determined by the dutch corpus as dutch tends to use more words than the english equivalent
#corpus_short = corpus.filter(items=idx_short_sentences, axis=0)
#sample = corpus_short.sample(round(len(corpus_short) * 0.03))
sample = corpus.sample(round(len(corpus) * 0.01))

eng_sentences = np.array(sample.eng)
nl_sentences = np.array(sample.nl)

# pre-pprocessing (task 2)
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

# Analysis of sample
print('Maximum length dutch sentence: {}'.format(len(max(nl_text_tokenized_word,key=len))))
print('Maximum length english sentence: {}'.format(len(max(eng_text_tokenized_word,key=len))))

# Check language length
nl_vocab = len(nl_text_tokenizer_word.word_index) + 1
eng_vocab = len(eng_text_tokenizer_word.word_index) + 1
nl_char = len(eng_text_tokenizer_char.word_index) + 1
eng_char = len(eng_text_tokenizer_char.word_index) + 1
print("Dutch vocabulary is of {} unique words".format(nl_vocab))
print("English vocabulary is of {} unique words".format(eng_vocab))
print("Dutch vocabulary is of {} characters".format(nl_char))
print("English vocabulary is of {} characters".format(eng_char))

# Respective max sentence length
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



# function for building ecoder-decoder model without embedding
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



# function for building ecoder-decoder model with Keras' embedding model
def keras_embd_encdec_model(input_shape, output_max_len, output_vocab_size):
    model = Sequential()
    model.add(Embedding(output_vocab_size, 64, input_length=input_shape[1])) #embeddings_initializer: Initializer for the embeddings matrix (see keras.initializers)
    model.add(GRU(64, return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
eng_pad_sentence_word.shape
nl_pad_sentence_word.shape
max_nl_len_word
max_eng_len_word
eng_vocab
nl_vocab
# build word-based english to dutch and dutch to english translator with keras embedding
eng2nl_keras_embd_encdec_word_model = keras_embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab)
nl2eng_keras_embd_encdec_word_model = keras_embd_encdec_model(nl_pad_sentence_word.shape, max_eng_len_word, eng_vocab)

# build character-based english to dutch and dutch to english translator with keras embedding
eng2nl_keras_embd_encdec_char_model = keras_embd_encdec_model(eng_pad_sentence_char.shape, max_nl_len_char, nl_char)
nl2eng_keras_embd_encdec_char_model = keras_embd_encdec_model(nl_pad_sentence_char.shape, max_eng_len_char, eng_char)





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
for word, i in eng_text_tokenizer_word.word_index.items():        
    embedding_vector = glove_word_vectors.get(word)
    if embedding_vector is not None: # non exisiting words will be zero            
        glove_embedding_matrix[i] = embedding_vector

# function for building ecoder-decoder model with Keras' embedding model
def glove_embd_encdec_model(input_shape, output_max_len, output_vocab_size):
    #build model
    model = Sequential()
    model.add(Embedding(glove_embedding_matrix.shape[0], 300, input_length=input_shape[1], embeddings_initializer=Constant(glove_embedding_matrix)))
    model.add(GRU(64, return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build models capable of using glove
eng2nl_glove_embd_encdec_word_model = glove_embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab)

#other pretrained weights: https://code.google.com/archive/p/word2vec/

model_training_manuals = [
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
        "title" : "English to Dutch translator (word-based, Glove embedding)",
        "model" : eng2nl_glove_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    },
    {   
        "title" : "English to Dutch translator (word-based, Keras' embedding)",
        "model" : eng2nl_keras_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    },
    {
        "title" : "Dutch to English translator (word-based, Keras' embedding)",
        "model" : nl2eng_keras_embd_encdec_word_model,
        "X" : nl_pad_sentence_word,
        "y" : eng_pad_sentence_word
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
    },
]


for manual in model_training_manuals:
    print("\t\t\t")
    print("MODEL: "+manual["title"])
    
    # show model architecture
    manual["model"].summary()

    # train and test model
    history = manual["model"].fit(manual["X"], manual["y"], batch_size=16, epochs=3, validation_split=0.2)

    # accuracy of last epoch
    print("Last measured accuracy score: ", history.history["accuracy"][2])

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