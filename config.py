train_path='shakespeare.txt'

#reading the data from the text file
text = open(train_path, 'r').read()
#print(text[:500])


# The unique characters in the file
vocab = sorted(set(text))
#print(vocab)
length=len(vocab)
#print(length)



#declaring the constants for the RNN model
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embed_dim = 64

# Number of RNN units
rnn_neurons = 1026