#The objective of the project is to implement language translation model aka machine translation for converting
#German to English (and vice versa) For this, the data is a text file (.txt) of 150,000 English-German sentence pairs. 
#But we use only the first 50,000 sentence pairs to reduce training time.
#first we import the required libraries.
# then we preprocess our text. this includes tokenization, cleaning(removing whitesp, punct,convert lowercase) 
# then we proceed to text2seq conversion(encode and decode)
#for example if we want to translate from eng to ger, the inputs to the encoder will be english sentences and outputs will be german sentences
#from the decoder which is the translation of the english sentences.
import matplotlib.pyplot as plt
import pandas as pd
import string
import re
from numpy import array, argmax, random, take
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 200)

# function to read raw text file
def read_text(filename):
    # open the file
    #mode is either read or write and text or binary
    #utf=uniform transformation format, here 8 bit values are used in the encoding
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    file.close()
    return text

# split a text into sentences we use split 
#here strip is used to remove the white space to the left and right of the string and 
# split breaks down a string into a list where each word is an item
#so, first we split the text into English-German pairs separated by newline
#we then split these pairs into English and German sentences respectively. 
def to_lines(text):
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    return sents

#read only the 0 to 50000th word in the file
data = read_text("deu.txt")
deu_eng = to_lines(data)
#we represent the data as an array(basically the text/data is converted into array/numbers)
deu_eng = array(deu_eng)
deu_eng = deu_eng[:50000:]
deu_eng

# To remove punctuation 
deu_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,0]]
deu_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,1]]
deu_eng
# convert to lowercase
for i in range(len(deu_eng)):
    deu_eng[i,0] = deu_eng[i,0].lower()
    
    deu_eng[i,1] = deu_eng[i,1].lower()
deu_eng
# empty lists
eng_l = []
deu_l = []

# populate the lists with sentence lengths
for i in deu_eng[:,0]:
    eng_l.append(len(i.split()))

for i in deu_eng[:,1]:
    deu_l.append(len(i.split()))
length_df = pd.DataFrame({'eng':eng_l, 'deu':deu_l})

length_df = pd.DataFrame({'eng':eng_l, 'deu':deu_l})
length_df.hist(bins = 30)

#we plot 2 different graphs for each of the languages 
# output- we infer that the maximum length of the german sentences is 11 and english 8.
plt.show()

# function to build a tokenizer
#tf.keras.preprocessing.text.Tokenizer allows to vectorize a text corpus, by turning each text into either a sequence of integers
#so sequences that are split into tokens are basically indexed or vectorized
#we transform sentences into sequences of integers
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
# prepare english tokenizer
#creating a vocabulary of all english sentences
eng_tokenizer = tokenization(deu_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('English Vocabulary Size: %d' % eng_vocab_size)
#o/p=English Vocabulary Size: 6453

# prepare Deutch tokenizer
#creating a vocabulary of all german sentences
deu_tokenizer = tokenization(deu_eng[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1

deu_length = 8
print('Deutch Vocabulary Size: %d' % deu_vocab_size)
#o/p=German Vocabulary Size: 10219

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    #this is done to make all sequences of the same length and appending 0s does not change the value or meaning of our initial sequence.
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

#for building the model using sklearn
#we split the entire data into 2 datasets, 1 for training and 1 for testing 
from sklearn.model_selection import train_test_split
#test_size=0.2 means that 20% of the data will be used for testing and the remaining 80% is used for training
train, test = train_test_split(deu_eng, test_size=0.2, random_state = 12)

# encode training data
#slicing is used to specify the intervals for training and testing
trainX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
# prepare validation data
testX = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

#Applying Seq2Seq model architecture to build Neural Machine Translational model
#here we use an embedding layer and an LSTM layer as our encoder and another LSTM with a dense layer for the decoder
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    #The softmax activation function transforms the raw outputs of the neural network into a vector of probabilities, 
    # essentially a probability distribution over the input classes.
    #reason why we use softmax act func over sigmoid, relu etc is bc range will 0 to 1, and the sum of all probabilities will be equal to one. 
    # If the softmax function used for multi-classification model it returns the probabilities of each class and
    # the target class will have the high probability
    model.add(Dense(out_vocab, activation='softmax'))
    return model

#we use rmsprop optimizer since it has a higher accuracy or rather a better optimization 
# (trains the model in different adaptive learning rates)
model = build_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 512)
#learning rate of 0.001 is used (neither too high nor too low) to reach global minima(stochastic gradient descent) in optimal time
rms = optimizers.RMSprop(lr=0.001)
#sparse_categorical_crossentropy is used here  because it allows us to use the huge target sequence as it is instead of one hot encoded format.
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

#we train it for 30 epochs although you can start with 5epochs if 30 takes too long depending on your system parameters
# we use model checkpoint() to save the best model with least validation loss.
filename = 'model.h1.20_feb_22'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 
          epochs=30, batch_size=512, 
          validation_split = 0.2,
          callbacks=[checkpoint], verbose=1)
#here we just plot 2 graphs to see the trend training and validation losses and we notice that with each eopoch, the loss decreases.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()

#we then load the model to make predictions.
model = load_model('model.h1.path')
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None
# convert predictions into text (English)
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
             
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t)            
        
    preds_text.append(' '.join(temp))
#Finally put the original English sentences in the test dataset and the predicted sentences in a data frame.
pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
pd.set_option('display.max_colwidth', 200)
#here we display the first 15 rows and last 15 rows of our dataset. If we increase the number of epochs, the model will learn better 
#and give more accurate predictions.
pred_df.head(15)
pred_df.tail(15)
pred_df.sample(15)
