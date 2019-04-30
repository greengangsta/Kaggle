#Importing the libraries
import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense,Embedding,Input,LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam,SGD

#Declaring the parameters
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 2000
LATENT_DIM =25

#Loading the training data
input_texts = []
target_texts = []

for line in open('robert_frost.txt'):
	line = line.rstrip()
	if line not in line:
		continue
	input_line = '<sos> ' + line
	target_line = line + ' <eos>'
	input_texts.append(input_line)
	target_texts.append(target_line)
	
	
#Data pre-processing	
all_lines = input_texts + target_texts
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE,filters ='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_sequence_length_from_data = max(len(s) for s in input_sequences)
print('Max sequence length : ', max_sequence_length_from_data)

word2idx = tokenizer.word_index
print('Found %s unique tokens. ' % len(word2idx))
assert('<sos>' in word2idx)
assert('<eos>' in word2idx)

max_sequence_length = max(max_sequence_length_from_data,MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences,maxlen=max_sequence_length,padding='post')
target_sequences = pad_sequences(target_sequences,maxlen=max_sequence_length,padding='post')
print('Shape of data tensor : ',input_sequences.shape)

