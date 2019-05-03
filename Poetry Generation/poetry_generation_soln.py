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
EMBEDDING_DIM = 100
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

#Loading the word vectors from stanford glove 6B 100D
word2vec={}
with open(os.path.join('glove.6B.100d.txt'),encoding="utf-8") as f:
	for line in f:
		values=line.split()
		word = values[0]
		vec = np.asarray(values[1:],dtype='float32')
		word2vec[word]=vec
	print('Found %s word vectors.'%len(word2vec))


# Creating the embedding matrix
print('Filling the embedding matrix...')
num_words = min(MAX_VOCAB_SIZE,len(word2idx))
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for words,i in word2idx.items():
	if i < MAX_VOCAB_SIZE:
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None :
			embedding_matrix[i] = embedding_vector
			
# Creating one-hot targets
one_hot_targets = np.zeros((len(input_sequences),max_sequence_length,num_words))
for i,target_sequence in enumerate(target_sequences):
	for t,word in enumerate(target_sequence):
		if word>0 :
			one_hot_targets[i,t,word] =1
			

#Building the model
embedding_layer = Embedding(num_words,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False)
print('Building the model....')
input_ = Input(shape = (max_sequence_length,))
initial_h = Input(shape=(LATENT_DIM,))
initial_c = Input(shape=(LATENT_DIM,))
x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM,return_sequences = True,return_state=True)
x,_,_ = lstm(x,initial_state=[initial_h,initial_c])
dense = Dense(num_words,activation='softmax')
output= dense(x)

model = Model([input_,initial_h,initial_c],output)
model.compile(loss = 'categorical_crossentropy',optimizer=Adam(lr=0.01),metrics = ['accuracy'])

#Training the model
print('Training the model....')
z = np.zeros((len(input_sequences),LATENT_DIM))
r = model.fit([input_sequences,z,z],
			  one_hot_targets,
			  batch_size = BATCH_SIZE,
			  epochs = EPOCHS,
			  validation_split = VALIDATION_SPLIT)

plt.plot(r.history['loss'],label='loss')
plt.plot(r,history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'],label='acc')
plt.plot(r,history['val_acc'],label='val_acc')
plt.legend()
plt.show()


#Building the sampling model 
input2 = Input(shape=(1,))
x = embedding_layer(input2)
x,h,c = lstm(x,initial_state = [initial_h,initial_c])
output2 = dense(x)
sampling_model = Model([input2,initial_h,initial_c],[output2,h,c])
idx2word = {v:k for k,v in word2idx.items()}


def sample_line():
	np_input = np.array([[word2idx['<sos>']]])
	h = np.zeros((1,LATENT_DIM))
	c = np.zeros((1,LATENT_DIM))
    
	eos = word2idx['<eos>']
	output_sentence =[]
	for _ in range (max_sequence_length):
		o,h,c = sampling_model.predict([np_input,h,c])
		probs = o[0,0]
		if np.argmax(probs)==0:
			print("wtf")
		probs[0] = 0
		probs/=probs.sum()
		idx = np.random.choice(len(probs),p=probs)
		if idx == eos:
			break
		output_sentence.append(idx2word.get(idx,'<WTF%s>'%idx))
		np_input[0,0] = idx
	return ' '.join(output_sentence)
     
while True :
	for _ in range(4):
		print(sample_line())
	ans = input("______generate another? [y/n]_______")
	if ans and ans[0].lower().startswith('n'):
		break