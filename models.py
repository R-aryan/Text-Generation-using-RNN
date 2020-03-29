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



