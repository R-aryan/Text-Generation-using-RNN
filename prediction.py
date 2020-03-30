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

print("\n \n model created successfully in prediction file and here is the summary of the model.....\n")

print(model.summary())



print("\n\n")
#code for generating text

def generate_text(model, start_seed,gen_size=100,temp=1.0):
  '''
  model: Trained Model to Generate Text
  start_seed: Intial Seed text in string form
  gen_size: Number of characters to generate

  Basic idea behind this function is to take in some seed text, format it so
  that it is in the correct shape for our network, then loop the sequence as
  we keep adding our own predicted characters. Similar to our work in the RNN
  time series problems.
  '''

  # Number of characters to generate
  num_generate = gen_size

  # Vecotrizing starting seed text
  input_eval = [char_to_ind[s] for s in start_seed]

  # Expand to match batch format shape
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty list to hold resulting generated text
  text_generated = []

  # Temperature effects randomness in our resulting text
  # The term is derived from entropy/thermodynamics.
  # The temperature is used to effect probability of next characters.
  # Higher probability == lesss surprising/ more expected
  # Lower temperature == more surprising / less expected
 
  temperature = temp

  # Here batch size == 1
  model.reset_states()

  for i in range(num_generate):

      # Generate Predictions
      predictions = model(input_eval)

      # Remove the batch shape dimension
      predictions = tf.squeeze(predictions, 0)

      # Use a cateogircal disitribution to select the next character
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # Pass the predicted charracter for the next input
      input_eval = tf.expand_dims([predicted_id], 0)

      # Transform back to character letter
      text_generated.append(ind_to_char[predicted_id])

  return (start_seed + ''.join(text_generated))


print(generate_text(model,"flower",gen_size=1000))
