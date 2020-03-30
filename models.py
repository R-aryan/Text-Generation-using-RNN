#importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cf


#importing tenserflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model



#defining modelfunction

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




# print("model created successfully")


#creating the model
model = create_model(cf.vocab_size, cf.embed_dim,cf. rnn_neurons, batch_size=1)

model.load_weights(cf.model_path)

model.build(tf.TensorShape([1, None]))

print("\n \n model created successfully in models file file and here is the summary of the model.....\n")

