import os
import sys
import numpy as np
import pandas as pd
#import matplotlib.pylplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,GlobalMaxPooling1D
from keras.layers import Conv1D,MaxPooling1D,Embedding
from keras.models import Model
from sklearn.metrics import ruc_auc_score


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT=0.2
BATCH_SIZE = 128
EPOCHS = 10

print('Loading word vectors...')

word2vec = {}

with open(os.path.join('glove.6B/glove.6B.100d.txt'),encoding="utf-8") as f:
	for line in f:
		values = line.split()
		word = values[0]
		vec = np.asarray(values[1:],dtype = 'float32')
		word2vec[word] = vec
		print('found %s word vectors.' %len(word2vec))
		
print('Loading the comments...')

train = pd.read_csv('train.csv')
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
targets = train[possible_labels].values

tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word2idx = tokenizer.word_index

print('found %s unique tokens.' %len(word2idx))

data = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor : ',data.shape)

print('Filling pre trained embeddings.....')
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)

embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))

for word,i in word2idx.items():
	if i < MAX_VOCAB_SIZE :
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None :
			embedding_matrix[i] = embedding_vector


		