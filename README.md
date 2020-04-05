# Text Generation with Neural Networks

In this repository we will create a network that can generate text, here we show it being done character by character.

 A very awesome write up on text generation: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

## About The Data

You can grab any free text you want from here: https://www.gutenberg.org/

We'll choose all of shakespeare's works (which we have already downloaded for you), mainly for two reasons:

1. Its a large corpus of text, its usually recommended you have at least a source of 1 million characters total to get realistic text generation.

2. It has a very distinctive style. Since the text data uses old style english and is formatted in the style of a stage play, it will be very obvious to us if the model is able to reproduce similar results.


## Training the Neural Network
For Training purpose-

- Clone The Repository by git clone https://github.com/R-aryan/Text-Generation-using-RNN.git
- Navigate to the **Training Directory**
- Open the **Text_generation_using_RNN.ipynb** Notebook
- Since the Notebbok is trained on colab the dataset/training path is according to colab directory structure so that should be changed accordingly.
- Run the notebbok from the start, Training will take some time. After Training is done the model will be saved as an **model_name.h5** file ,again the path where you want to save the model should be changed accordingly, before running the notebook.


## Inference

For Inference purposem we will use the saved model or saved model can also be downloaded from here-
- The model/weights can be downloaded from here :- [Download Weights/Model](https://drive.google.com/open?id=1-346UfIYLVMRXU3tKY_euip9u2mLRTG9)
- After downloading/saving the trained model navigate to the **inference folder and open the config.py file.**
- Change the value of **model_path variable in config.py file**  according to the directory where the saved model is stored.

