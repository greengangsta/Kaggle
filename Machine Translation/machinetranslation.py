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