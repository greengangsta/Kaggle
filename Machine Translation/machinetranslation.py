#importing the libraries

import os,sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,LSTM,GRU,Dense,Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


#Defining some parameters

BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100


input_texts = []
target_texts = []
target_texts_inputs = []

#Loading the data
t = 0
for line in open('hin.txt',encoding='UTF-8'):
	t+=1
	if t>NUM_SAMPLES:
		break
	if '\t' not in line:
		continue
	input_text,translation = line.split('\t')
	
	target_text = translation + ' <eos>'
	target_text_input = '<sos> ' + translation
	input_texts.append(input_text)
	target_texts.append(target_text)
	target_texts_inputs.append(target_text_input)
print("num samples : ",len(input_texts))


# tokenizing the inputs
tokenizer_inputs = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens. '%len(word2idx_inputs))
max_len_input = max(len(s) for s in input_sequences)

#tokenize the outputs
tokenizer_outputs = Tokenizer(num_words = MAX_NUM_WORDS,filters ='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens. ' %len(word2idx_outputs))

num_words_output = len(word2idx_outputs) +1 
max_len_target = max(len(s) for s in target_sequences)

#padding the sequences 
encoder_inputs = pad_sequences(input_sequences,maxlen = max_len_input)
print("encoderdata.shape : ",encoder_inputs.shape)
print("encoder_data[0] : " , encoder_inputs[0])
decoder_inputs = pad_sequences(target_sequences_inputs,maxlen = max_len_target,padding='post')
print("deoderdata.shape : ",decoder_inputs.shape)
print("decoder_data[0] : " , decoder_inputs[0])

decoder_targets = pad_sequences(target_sequences,maxlen = max_len_target,padding='post')

#loading the pretrained vectors
print('Loading the vectors...')
word2vec = {}
with open('glove.6B.100d.txt',encoding='utf-8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		vec = np.asarray(values[1:],dtype = 'float32')
		word2vec[word] = vec
print('Found %s word vectors. ' %len(word2vec))


#preparing the embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS,len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx_inputs.items() :
	if i < MAX_NUM_WORDS :
		embedding_vector = word2vec.get(word)
	if embedding_vector is not None :
		embedding_matrix[i] = embedding_vector

#vreating the embedding layer
embedding_layer = Embedding(num_words,EMBEDDING_DIM,weights = embedding_matrix,input_length=max_len_input)

# creating one-hot encoded targets
decoder_targets_one_hot = np.zeros((len(input_texts),max_len_target,num_words_output),dtype = 'float32')