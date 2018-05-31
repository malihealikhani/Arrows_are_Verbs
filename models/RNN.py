import pandas as pd
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


all_data_train = pd.read_csv('./fold3/train3')
all_data_test = pd.read_csv('./fold3/test3')

X1= all_data_train["sentences"]
y1= all_data_train["label"]

X2= all_data_test["sentences"]
y2= all_data_test["label"]

elements = (' '.join([sentence for sentence in X1])).split()
X_train, X_test = X1, X2
y_train, y_test = y1, y2
categories = set(y1)

def create_lookup_tables(text):
    """Create lookup tables for vocabulary """
    vocab = set(text)
    
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {v:k for k, v in vocab_to_int.items()}
    
    return vocab_to_int, int_to_vocab


elements.append("<UNK>")
vocab_to_int, int_to_vocab = create_lookup_tables(elements)
categories_to_int, int_to_categories = create_lookup_tables(y1)
print("vocabulary of the dataset: {}".format(len(vocab_to_int)))



def convert_to_int(data, data_int):
    """Converts all text to integers"""
    all_items = []
    for sentence in data: 
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])
    return all_items


X_test_encoded = convert_to_int(X_test, vocab_to_int)
X_train_encoded = convert_to_int(X_train, vocab_to_int)
y_data = convert_to_int(y_test, categories_to_int)


enc = OneHotEncoder()
enc.fit(y_data)

y_train_encoded = enc.fit_transform(convert_to_int(y_train, categories_to_int)).toarray()
y_test_encoded = enc.fit_transform(convert_to_int(y_test, categories_to_int)).toarray()

max_sentence_length = 20
embedding_vector_length = 64
dropout = 0.3


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


X_train_pad = sequence.pad_sequences(X_train_encoded, maxlen=max_sentence_length)
X_test_pad = sequence.pad_sequences(X_test_encoded, maxlen=max_sentence_length)

model = Sequential()
model.add(Embedding(len(vocab_to_int), embedding_vector_length, input_length=max_sentence_length))
model.add(Activation('relu'))

model.add(LSTM(64, dropout=dropout, recurrent_dropout=dropout))
model.add(Dense(len(categories), activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', mean_pred])

history= model.fit(X_train_pad, y_train_encoded, epochs=5, batch_size=16)
scores = model.evaluate(X_test_pad, y_test_encoded, verbose=0)




def predict_sentence(sentence):
    """converts the text and sends it to the model for classification."""

    x = np.array(convert_to_int([sentence], vocab_to_int))
    x = sequence.pad_sequences(x, maxlen=max_sentence_length)
    
    prediction = model.predict(x)

    lang_index = np.argmax(prediction)
    
    return int_to_categories[lang_index]
    