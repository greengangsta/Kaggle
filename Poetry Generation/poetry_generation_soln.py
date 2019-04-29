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

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 2000
LATENT_DIM =25

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