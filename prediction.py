#to generate the text character by character after training the model


#importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cf
import models as md


#importing tenserflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model


vocab= cf.vocab
#print(cf.length)

#From the above output we can infer that the entire text corpus consists of 84 unique characters.

#We know a neural network can't take in the raw string data,we need to assign numbers to each character. 
# Let's create two dictionaries that can go from numeric index to character and character to numeric index.

char_to_ind = {u:i for i, u in enumerate(vocab)}
#print(char_to_ind)

#converting from index to character

ind_to_char = np.array(vocab)
#print(ind_to_char)


#creating the model
model = md.create_model(cf.vocab_size, cf.embed_dim,cf. rnn_neurons, batch_size=1)

model.load_weights(cf.model_path)

model.build(tf.TensorShape([1, None]))

print("model created successfully in prediction file and here is the summary of the model.....\n")

print(model.summary())

