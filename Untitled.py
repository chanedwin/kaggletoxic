
# coding: utf-8

# In[3]:


import csv
import h5py
import numpy as np


from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer

DATA_FILE = './data/train.csv'
W2V_MODEL = './models/w2v.840B.300d.txt'


# In[4]:


from gensim.models.keyedvectors import KeyedVectors


def load_w2v_model_from_path(model_path, binary_input=False):
    """
    :param model_path: path to w2v model
    :type model_path: string
    :param binary_input: True : binary input, False : text input
    :type binary_input: boolean
    :return: loaded w2v model
    :rtype: KeyedVectors object
    """
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=binary_input)
    return w2v_model


model = load_w2v_model_from_path(W2V_MODEL)


# In[5]:


full_data_set = []

with open(DATA_FILE) as f:
    reader = csv.reader(f)
    header = next(reader)
    for line in reader:
        full_data_set.append(line)

# load data into native lists
print(header)
id = [i for i in map(lambda x: x[0], full_data_set)]
text = [i for i in map(lambda x: x[1], full_data_set)]
toxic = [i for i in map(lambda x: x[2], full_data_set)]
severe_toxic = [i for i in map(lambda x: x[3], full_data_set)]
obscene = [i for i in map(lambda x: x[4], full_data_set)]
threat = [i for i in map(lambda x: x[5], full_data_set)]
insult = [i for i in map(lambda x: x[6], full_data_set)]
identity_hate = [i for i in map(lambda x: x[6], full_data_set)]


# In[6]:


tknzr = TweetTokenizer()
max_length = 0
#tokenize sentences
tokenized_sentences = []
for sentence in text:
    tokenized_sentences.append(tknzr.tokenize(sentence))
    max_length = max(max_length,len(sentence))


# In[1]:


#vectorise sentences
removed_indexes = []
vectorized_sentences = []
for i in range(len(tokenized_sentences)):
    tokenized_sentence = tokenized_sentences[i]
    if len(tokenized_sentence) > 30 :
        tokenized_sentence = tokenized_sentence[:30]
    vector_rep_of_sentence = []
    for word in tokenized_sentence:
        if word in model.vocab:
            vector_rep_of_sentence.append(model.wv[word])
    if not vector_rep_of_sentence :
        removed_indexes.append(i)
    else :
        array = np.array(vector_rep_of_sentence)
        zeroes = np.zeros((30-len(vector_rep_of_sentence),300))
        vector_rep_of_sentence = np.concatenate((array,zeroes),axis=0)
        vectorized_sentences.append(vector_rep_of_sentence)


# In[15]:


np_vectorized_sentences = np.array(vectorized_sentences)


# In[23]:


print(np.reshape(np_vectorized_sentences,(-1,140,300)))


# In[94]:


result = np.zeros((len(vectorized_sentences),140,300))


# In[95]:


for i in range(len(result)):
    result[i][:vectorized_sentences[i].shape[0],:vectorized_sentences[i].shape[1]] =vectorized_sentences[i]


# In[79]:


result[:vectorized_sentences.shape[0],:vectorized_sentences.shape[1],:vectorized_sentences.shape[2]] = vectorized_sentences


# In[13]:


new_vectorized_sentences = np.ndarray(vectorized_sentences)


# In[10]:


print(vectorized_sentences.shape)


# In[31]:


vectorized_sentences = np.pad(vectorized_sentences,(2048,),"constant")


# In[32]:


print(vectorized_sentences[:10])


# In[7]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 300
timesteps = 8
num_classes = 2

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))


# In[9]:





# In[5]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 300
timesteps = 8
num_classes = 2
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))


# In[68]:


import sys
if sys.maxsize > 2**32:
    print('64-bit')
else:
    print('32-bit')

