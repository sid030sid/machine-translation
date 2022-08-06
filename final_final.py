import string # for string manipulation during pre-processing
import pandas as pd # for data frames
import numpy as np # for calculations
import matplotlib.pyplot as plt # for plotting
from keras.models import Sequential # for building encoder decoder model with RNNs
from keras.layers import RepeatVector, GRU, Dense, TimeDistributed, Embedding # for building encoder decoder model with RNNs
from keras.preprocessing.text import Tokenizer # for pre-processing
from keras_preprocessing.sequence import pad_sequences # for pre-processing
from keras.initializers import Constant # for matrix with pre-trained embedding weights
from keras_self_attention import SeqSelfAttention as Attention # for attention mechanism
from gensim.models import KeyedVectors # to load in pretrained word2vec weights
import seaborn as sns # for creating boxplots
from collections import Counter # for further array functionality
import dataframe_image as dfi # to store data frames as images



# import corpus
eng = pd.read_csv("data/europarl-v7.nl-en.en", sep="\t", names=["eng"])
nl = pd.read_csv("data/europarl-v7.nl-en.nl", sep="\t", names=["nl"], on_bad_lines='skip') #bad lines are skiped due to pandas.errors.ParserError: Expected 1 fields in line 952982, saw 2
corpus = pd.DataFrame(eng)
corpus["nl"] = nl



# TASK 1: analysis of corpus
# array of all words in corpus
words_eng = [word for sentence in corpus.eng for word in sentence.split()]
words_nl = [word for sentence in corpus.nl for word in sentence.split()]

# array of sentences' length
num_word_per_sentence_eng = [len(sentence.split()) for sentence in corpus.eng]
num_word_per_sentence_nl = [len(sentence.split()) for sentence in corpus.nl]

# visualise sentence length
data = pd.concat([
    pd.DataFrame({"Number of words":num_word_per_sentence_nl, "Language":"Dutch"}),
    pd.DataFrame({"Number of words":num_word_per_sentence_eng, "Language":"English"})
])

boxplot = sns.boxplot(y = "Number of words", x="Language", data=data).set_title('Length of English and Dutch sentences')
fig = boxplot.figure.savefig("documentation/plots/summary_corpus/boxplot_number_words.png", format="png")
plt.clf()

# visualise top 10 most common used words
plt.bar([item[0] for item in Counter(words_eng).most_common(10)], [item[1] for item in Counter(words_eng).most_common(10)])
plt.title('Top 10 most used English words')  
plt.savefig("documentation/plots/summary_corpus/top10_eng_words.png", format="png")
plt.clf()

plt.bar([item[0] for item in Counter(words_nl).most_common(10)], [item[1] for item in Counter(words_nl).most_common(10)])
plt.title('Top 10 most used English words')  
plt.savefig("documentation/plots/summary_corpus/top10_nl_words.png", format="png")
plt.clf()

# store values for documentation
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
def get_suitable_sample(): 

    # due to limited resources select random 1% of corpus
    sample = corpus.sample(round(len(corpus) * 0.01)) 
    
    # due to limited CPU no outlier sentences in terms of number of words can be present in sample (otherwise too big input matrices for models)
    eng_max_num_word_per_sentence = max([len(sentence.split()) for sentence in sample.eng])
    nl_max_num_word_per_sentence = max([len(sentence.split()) for sentence in sample.nl])

    if eng_max_num_word_per_sentence > 500 or nl_max_num_word_per_sentence > 500:
        sample = get_suitable_sample() # if outliers are present re-call function till suitable sample found
    return sample

# due to limited resources use conditional random sampling
sample = get_suitable_sample() 

# transform english and dutch corpus into single numpy arrays for further steps
eng_sentences = np.array(sample.eng)
nl_sentences = np.array(sample.nl)



# TASK 2: pre-pprocessing
def clean_sentence(sentence):
    # lower case sentences
    lower_case_sent = sentence.lower()

    # remove punctuation
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
nl_char = len(nl_text_tokenizer_char.word_index) + 1
eng_char = len(eng_text_tokenizer_char.word_index) + 1

# summary of sample for documentation purposes
eng_avg_len = np.mean([len(sentence) for sentence in eng_text_tokenized_word]) # average amount of words in a sentence of englisch corpus
eng_num_word = sum([len(sentence) for sentence in eng_text_tokenized_word]) # number of words in english corpus
eng_max_len = len(max(eng_text_tokenized_word,key=len)) # maximum sentence lenght in english corpus
eng_min_len = len(min(eng_text_tokenized_word,key=len)) # minimum sentence lenght in english corpus

nl_avg_len = np.mean([len(sentence) for sentence in nl_text_tokenized_word]) # average amount of words in a sentence of dutch corpus
nl_num_word = sum([len(sentence) for sentence in nl_text_tokenized_word]) # number of words in dutch corpus
nl_max_len = len(max(nl_text_tokenized_word,key=len)) # maximum sentence lenght in dutch corpus
nl_min_len = len(min(nl_text_tokenized_word,key=len)) # minimum sentence lenght in dutch corpus

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



# TASK 3: 
# comparison of performance of neural machine translator depending word embedding (no embedding, Keras' embedding, GloVe, Word2Vec) and tokenization level (word or character)

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



# Keras' embedding model:
# function for building ecoder-decoder model with Keras' embedding model
def keras_embd_encdec_model(input_shape, output_max_len, output_vocab_size, input_vocab_size):
    model = Sequential()
    model.add(Embedding(input_vocab_size, 300, input_length=input_shape[1], trainable = True)) #output dimension is set to 300 in order to be comparable with fixed output dim of glove and word2vec embedding
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
# get all of glove's pre-trained weights for english words (source: https://nlp.stanford.edu/projects/glove/)
eng_glove_word_vectors = {}
with open("data\glove.6B.300d.txt", encoding="utf8") as glove_data:
    for element in glove_data:
        values = element.split()            
        word = values[0]            
        coefs = values[1:]            
        eng_glove_word_vectors[word] = np.asarray(coefs, dtype='float32')

# select glove vectors whose words are also present in the english corpus and store it in a matrix
eng_glove_embedding_matrix = np.zeros((eng_vocab, 300)) # 300 = word vector dimension = number of weights per word in glove
eng_words_not_in_glove = []
for word, i in eng_text_tokenizer_word.word_index.items():        
    embedding_vector = eng_glove_word_vectors.get(word)
    if embedding_vector is not None: # non exisiting words will be zero            
        eng_glove_embedding_matrix[i] = embedding_vector
    else : 
        eng_words_not_in_glove.append(word)



# word2vec embedding:
# get all of word2vec's pre-trained weights for english words (source: https://code.google.com/archive/p/word2vec/)
eng_word2vec = KeyedVectors.load_word2vec_format("data\GoogleNews-vectors-negative300.bin", binary=True)

# select word2vec vectors whose words are also present in the english corpus and store it in a matrix
eng_word2vec_embedding_matrix = np.zeros((eng_vocab, 300)) # 300 = word2vec vector dimension (= 300 weigths per word)
eng_words_not_in_word2vec = []
for word, i in eng_text_tokenizer_word.word_index.items():  
    if word in eng_word2vec : 
        eng_word2vec_embedding_matrix[i] = eng_word2vec[word]
    else : 
        eng_words_not_in_word2vec.append(word)

# get all of word2vec's pre-trained weights for dutch words (source: https://github.com/clips/dutchembeddings):
nl_word2vec = KeyedVectors.load_word2vec_format("data\wikipedia-320.txt") # get all pretrained weigths

# select word2vec vectors whose words are also present in the dutch corpus and store it in a matrix
nl_word2vec_embedding_matrix = np.zeros((nl_vocab, 320)) # 320 = word2vec vector dimension (= 320 weigths per word)
nl_words_not_in_word2vec = []
for word, i in nl_text_tokenizer_word.word_index.items():  
    if word in nl_word2vec : 
        nl_word2vec_embedding_matrix[i] = nl_word2vec[word]
    else : 
        nl_words_not_in_word2vec.append(word)

# function for building encoder-decoder model with pretrained embedding weights
def embd_encdec_model(input_shape, output_max_len, output_vocab_size, input_vocab_size, embedding_matrix):
    #build model
    model = Sequential()
    model.add(Embedding(input_vocab_size, embedding_matrix.shape[1], input_length=input_shape[1], embeddings_initializer=Constant(embedding_matrix)))
    model.add(GRU(64, return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build eng2nl and nl2eng machine translator based on words and using pre-trained glove and word2vec weights
eng2nl_glove_embd_encdec_word_model = embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab, eng_vocab, eng_glove_embedding_matrix)
eng2nl_word2vec_embd_encdec_word_model = embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab, eng_vocab, eng_word2vec_embedding_matrix)
nl2eng_word2vec_embd_encdec_word_model = embd_encdec_model(nl_pad_sentence_word.shape, max_eng_len_word, eng_vocab, nl_vocab, nl_word2vec_embedding_matrix)


# TASK 4: neural machine translation with attention
# function for building encoder-decoder model with pretrained embedding weights and attention mechanism (source:https://pypi.org/project/keras-self-attention/0.0.10/)
def attention_embd_encdec_model(input_shape, output_max_len, output_vocab_size, input_vocab_size, embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_vocab_size, embedding_matrix.shape[1], input_length=input_shape[1], embeddings_initializer=Constant(embedding_matrix)))
    model.add(GRU(64, return_sequences = False))
    model.add(RepeatVector(output_max_len))
    model.add(Attention())
    model.add(GRU(64, return_sequences = True))
    model.add(TimeDistributed(Dense(output_vocab_size, activation = 'softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build models based on words with attention mechanism and pre-trained glove and word2vec weights
eng2nl_attention_glove_embd_encdec_word_model = attention_embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab, eng_vocab, eng_glove_embedding_matrix)
eng2nl_attention_word2vec_embd_encdec_word_model = attention_embd_encdec_model(eng_pad_sentence_word.shape, max_nl_len_word, nl_vocab, eng_vocab, eng_word2vec_embedding_matrix)
nl2eng_attention_word2vec_embd_encdec_word_model = attention_embd_encdec_model(nl_pad_sentence_word.shape, max_eng_len_word, eng_vocab, nl_vocab, nl_word2vec_embedding_matrix)

# the essential models for task 3
# - eng2nl glove --> 
# - eng2nl word2vec -->
# - nl2eng word2vec --> weights: https://github.com/clips/dutchembeddings
# - eng2nl char based

#essential models for task 4
# - eng2nl word2vec with attention

model_training_manuals = [
    {   
        "title" : "Dutch to English translator (word-based, Word2Vec embedding)",
        "model" : nl2eng_word2vec_embd_encdec_word_model,
        "X" : nl_pad_sentence_word,
        "y" : eng_pad_sentence_word
    },
     
       {   
        "title" : "Dutch to English translator (word-based, Word2Vec embedding, with attention",
        "model" : nl2eng_attention_word2vec_embd_encdec_word_model,
        "X" : nl_pad_sentence_word,
        "y" : eng_pad_sentence_word
    },
    {   
        "title" : "English to Dutch translator (word-based, Word2Vec embedding, with attention)",
        "model" : eng2nl_attention_word2vec_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    },
    {   
        "title" : "English to Dutch translator (word-based, Glove embedding, with attention)",
        "model" : eng2nl_attention_glove_embd_encdec_word_model,
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
        "title" : "English to Dutch translator (word-based, Word2Vec embedding)",
        "model" : eng2nl_word2vec_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    },
    {
        "title" : "English to Dutch translator (char-based)", 
        "model" : eng2nl_encdec_char_model,
        "X" : eng_pad_sentence_char,
        "y" : nl_pad_sentence_char
    },
    {
        "title" : "Dutch to English translator (word-based)",
        "model" : nl2eng_encdec_word_model,
        "X" : nl_pad_sentence_word,
        "y" : eng_pad_sentence_word
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
    },
       {   
        "title" : "English to Dutch translator (word-based, Glove embedding)",
        "model" : eng2nl_glove_embd_encdec_word_model,
        "X" : eng_pad_sentence_word,
        "y" : nl_pad_sentence_word
    }, 
]


for manual in model_training_manuals:
    print("\t\t\t")
    print("MODEL: "+manual["title"])
    
    # show model architecture
    manual["model"].summary()

    # train and test model
    history = manual["model"].fit(manual["X"], manual["y"], batch_size=16, epochs=3, validation_split=0.2)

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
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("documentation/plots/loss/"+manual["title"]+".png", format="png")
    plt.clf()

#function for converting tokenized data to text
def toText(sentence, vocab):
    return ""
