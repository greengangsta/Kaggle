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
