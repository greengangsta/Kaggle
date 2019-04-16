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
#from sklearn.metrics import ruc_auc_score


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT=0.2
BATCH_SIZE = 128
EPOCHS = 10

print('Loading word vectors...')

word2vec = {}

with open(os.path.join('')) as f:
	for line in f:
		values = line.split()
		word = values[0]
		vec = np.asarray(values[1,:],dtype = 'float32')
		word2vec[word] = vec
		print('found %s word vectors. ',%len(word2vec))