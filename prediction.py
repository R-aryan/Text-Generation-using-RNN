#to generate the text character by character after training the model


#importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cf


#importing tenserflow libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model


vocab= cf.vocab
print(cf.length)

#From the above output we can infer that the entire text corpus consists of 84 unique characters.

#We know a neural network can't take in the raw string data,we need to assign numbers to each character. 
# Let's create two dictionaries that can go from numeric index to character and character to numeric index.

char_to_ind = {u:i for i, u in enumerate(vocab)}
print(char_to_ind)

#converting from index to character

ind_to_char = np.array(vocab)
print(ind_to_char)



def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)



def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()

    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))

    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))

    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))

    model.compile(optimizer='adam', loss=sparse_cat_loss) 
    
    return model


#creating the model
model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)

model.load_weights(cf.model_path)

model.build(tf.TensorShape([1, None]))

