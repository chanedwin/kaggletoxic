import csv
import h5py
import numpy as np


from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer

DATA_FILE = './data/train.csv'
W2V_MODEL = './models/w2v.840B.300d.txt'

full_data_set = []

with open(DATA_FILE) as f:
    reader = csv.reader(f)
    header = next(reader)
    for line in reader:
        full_data_set.append(line)

print(header)
id = [i for i in map(lambda x: x[0], full_data_set)]
text = [i for i in map(lambda x: x[1], full_data_set)]
toxic = [i for i in map(lambda x: x[2], full_data_set)]
severe_toxic = [i for i in map(lambda x: x[3], full_data_set)]
obscene = [i for i in map(lambda x: x[4], full_data_set)]
threat = [i for i in map(lambda x: x[5], full_data_set)]
insult = [i for i in map(lambda x: x[6], full_data_set)]
identity_hate = [i for i in map(lambda x: x[6], full_data_set)]


# load glove models

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
tknzr = TweetTokenizer()

tokenized_sentences = []
for sentence in text:
    tokenized_sentences.append(tknzr.tokenize(sentence))

vectorized_sentences = []
for tokenized_sentence in tokenized_sentences:
    vector_rep_of_sentence = []
    for word in tokenized_sentence:
        if word in model.vocab:
            vector_rep_of_sentence.append(model.wv[word])
    vectorized_sentences.append(vector_rep_of_sentence)

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('dataset_1', data=vectorized_sentences)


h5f.close()