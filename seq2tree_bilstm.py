

"""""""""
Encoders basic description:

Say that you have hidden vectors
f1, f2, f3 and f4 (corresponding to "the brown dog eats") for the forward model and
b1, b2, b3, b4 (corresponding to "eats dog brown the") for the backward model. 
Then you either:

- take the last: 
you use the concatenation of (f4 and b4).

- take the maximum: 
you compute max(f1,f2,f3,f4) component-wise
(the same for the forward and for the backward),
thus getting a vector of the same dimensionality of f_i, and also max(b1,b2,b3,b4) and,
again you concatenate them.

Then you feed the vectors obtained in this way to a 
linear classifier to train the probing tasks.

"""""

##############################################

# Imports

#############################################

from keras import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.layers import Reshape
from pylab import *
from corpora_tools import *


##############################################

# Corpus and vectorization

#############################################

# SpaCy lineBYline pipeline
it_nlp = spacy.load('it_core_news_sm')
en_nlp = spacy.load('en_core_web_sm')

ln = file_len(ep_en)

nlp = en_nlp
files = ep_en
trees = []
sents = []
file = open(files, 'r')
line = file.readline()
for line in file:
    doc = nlp(line)
    # tokenize sentences
    sents.append(line)
    # dependency trees sent)
    trees.append([list(to_nltk_tree(sn.root)) for sn in doc.sents])
# transforms trees in integers sequence, eos is the highest integers
dep_trees, eos = clean_tree(trees, get_eos=True)

# visualizing obtained data
max_len(dep_trees, box_plot=True)

# vectorization (with corpus cleaning and all)
lang_model = sents2space(sents, 'en', 300, 0)


# setting the network corpus
s2t_network_corpus = s2t_trainig_set(cp, dep_trees, lang_model, eos)

# Training data set
network_corpus = s2t_network_corpus
train_size = int((len(network_corpus)*65)/100)
class_size = eos+1
x_size = len(network_corpus[0][1])
y_size = len(network_corpus[0][0])
x = list(np.zeros(train_size))
y = list(np.zeros(train_size))
for i in range(train_size):
    x[i] = [vec for vec in network_corpus[i][1]]
    y[i] = network_corpus[i][0]
    # reshape: sample, time steps, feature at each time step.
    # if I have 1000 sentences of 10 words, presented in a 3-dim vector:
    # is nb_samples = 1000, time steps =  10, input_dim = 3
X = array(x).reshape(train_size, x_size, 300)  # reshapes date into 3D matrix
Y = array(y).reshape(train_size, y_size, 1)  # reshapes date into 3D matrix


##############################################

# Max/Lastencoders models

#############################################

# x_size = 3
# y_size = 3
# class_size = 3

# shared input
inputs = Input(shape=(None, 300))


'# MAX_ENCODER #'
# Forward LSTM
# input_shape = (time_steps, features)
f_lstm = LSTM(512, return_sequences=True)(inputs)

# Forward Max-pooling  component-wise
max_f = GlobalMaxPooling1D()(f_lstm)

# backward LSTM
# input_shape = (time_steps, features)
b_lstm = LSTM(512, return_sequences=True, go_backwards=True)(inputs)

# Backwards Max-pooling  component-wise
max_b = GlobalMaxPooling1D()(b_lstm)

# Concatenate (extract here the output after training, using command)
concatenation = Concatenate(axis=-1)([max_f, max_b])

# to create a dim= representation
dense = Dense(300)(concatenation)

# reshape into a 3D tensor list (add time_steps=1)
reshape = Reshape((1, 300), input_shape=(x_size, 300))(dense)

# decoder (theoretically is a one-to-many, practically is a many-to-many)
outputs = LSTM(y_size, return_sequences=True)(reshape)

r2 = Reshape((y_size, 1), input_shape=(x_size, y_size))(outputs)

out = Dense(class_size, activation='sigmoid')(r2)

# d2 = TimeDistributed(Dense(101, activation='sigmoid'))(outputs)

# Compiling the model
max_model = Model(inputs=inputs, outputs=out)
max_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
max_model.summary()
max_model.fit(X, Y, epochs=3)


'# LAST_ENCODER #'
# Forward LSTM
# input_shape = (time_steps, features)
# bi_lstm = Bidirectional(LSTM(512, return_sequences=False),
#                         merge_mode='concat')(inputs)

# last_dense = Dense(300)(bi_lstm)

# r1 = Reshape((1, 300), input_shape=(x_size, 300))(last_dense)

# last_lstm = LSTM(y_size, return_sequences=True)(r1)

# r2 = Reshape((y_size, 1), input_shape=(x_size, y_size))(last_lstm)

# out = Dense(class_size, activation='sigmoid')(r2)

# # Compiling the model
# last_model = Model(inputs=inputs, outputs=out)
# last_model.compile(optimizer='adam',
#                    loss='sparse_categorical_crossentropy',
#                    metrics=['accuracy'])
# last_model.summary()
# last_model.fit(X, Y, epochs=3)


##############################################

# Test/Training

#############################################

# Testing
to_test = len(network_corpus)-train_size
predictions = list(np.zeros(to_test))
correct = list(np.zeros(to_test))
vi_correct = 0
for e in range(to_test):
    print('Prediction'+str(e)+'/'+str(to_test))
    test = []
    test = [v for v in network_corpus[e+train_size][1]]
    correct[e] = network_corpus[e+train_size][0]
    X = array(test).reshape(1, x_size, 300)
    resp = max_model.predict(X, verbose=2)  # gives probability distributions
    predictions[e] = resp.argmax(axis=-1)  # extract the actual predicted sequence
    print('Completed')

##############################################

# Hidden representation extraction

#############################################

# Hidden max-representation extraction
layer_name = 'hidene_layer_name'  # use .summary() to extract the name of the layer
hidden_max = Model(inputs=max_model.input,
                   outputs=max_model.get_layer(layer_name).output)
intermediate_output = hidden_max.predict(X)




