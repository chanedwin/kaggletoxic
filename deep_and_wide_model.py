import os
import pickle
import time

import keras.callbacks
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing import sequence

from utils import TOXIC_TEXT_INDEX, TRUTH_LABELS, COMMENT_TEXT_INDEX
from utils import load_data, vectorise_tweets, tokenize_tweets

from keras import backend as K

MAX_BATCH_SIZE_PRE_TRAINED = 400
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))


MAX_NUM_WORDS_ONE_HOT = 50000

FILE_NAME_STRING_DELIMITER = "_"
FILE_NAME_STRING_FORMATING = "%d_%m_%y_%H:%M"
KERAS_MODEL_DIRECTORY = 'keras_models/{}'
TRAIN_HISTORY_DICT_PATH = 'keras_models/{}/trainHistoryDict'
MODEL_SAVE_PATH = 'keras_models/{}/keras_model.h5'

MAXLEN = 3000

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

USE_WORD2VEC = True


def main(data_file, w2v_model, testing, expt_name="test"):
    if testing:
        print("running tests")
        number_of_epochs = 1
    else:
        print("running eval")
        number_of_epochs = 1

    # load data
    print("loading data")
    df = load_data(data_file)

    # process data
    print("processing data")
    if USE_WORD2VEC:
        tokenize_tweets(df)
        np_text_array = _convert_to_w2v_encoding(df, w2v_model)
        truth_dictionary = extract_truth_labels_as_dict(df)
        data_dict = split_train_test(np_text_array, truth_dictionary)
        key = TOXIC_TEXT_INDEX  # testing with 1 key for now
        x_train = data_dict[key][X_TRAIN_DATA_INDEX]
        padded_x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
        y_train = data_dict[key][Y_TRAIN_DATA_INDEX]

        x_test = data_dict[key][X_TEST_DATA_INDEX]
        padded_x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
        y_test = data_dict[key][Y_TEST_DATA_INDEX]

        # build neural network model
        model = build_keras_model()
        print("training network")
        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

        history = model.fit(padded_x_train, y_train,
                            batch_size=1024, epochs=number_of_epochs,
                            validation_data=(padded_x_test, y_test),
                            callbacks=[early_stop_callback, ]
                            )

        # try some values
        x_predict = model.predict(padded_x_test[::1000])
        for index, val in enumerate(x_predict):
            print("predicted is {}, truth is {},".format(x_predict[index][0], y_train[index]))
        save_model_details_and_training_history(expt_name, history, model)
    else:
        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_ONE_HOT,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=" ",
                              char_level=False)

        tokenizer.fit_on_texts(df[COMMENT_TEXT_INDEX])
        transformed_text = tokenizer.texts_to_sequences(df[COMMENT_TEXT_INDEX])
        vocab_size = len(tokenizer.word_counts)

        print("vocab length is", len(tokenizer.word_counts))
        model = build_keras_embeddings_model(vocab_size)
        truth_dictionary = extract_truth_labels_as_dict(df)

        data_dict = split_train_test(transformed_text, truth_dictionary)

        key = TOXIC_TEXT_INDEX  # testing with 1 key for now
        x_train = data_dict[key][X_TRAIN_DATA_INDEX]
        padded_x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
        y_train = data_dict[key][Y_TRAIN_DATA_INDEX]

        x_test = data_dict[key][X_TEST_DATA_INDEX]
        padded_x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
        y_test = data_dict[key][Y_TEST_DATA_INDEX]

        # build neural network model
        print("training network")
        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

        history = model.fit(padded_x_train, y_train,
                            batch_size=MAX_BATCH_SIZE_PRE_TRAINED, epochs=number_of_epochs,
                            validation_data=(padded_x_test, y_test),
                            callbacks=[early_stop_callback, ]
                            )

        # try some values
        x_predict = model.predict(padded_x_test[::1000])
        for index, val in enumerate(x_predict):
            print("predicted is {}, truth is {},".format(x_predict[index][0], y_train[index]))


def _convert_to_w2v_encoding(df, w2v_model):
    vectorise_tweets(w2v_model, df)
    drop_words_with_no_vectors_at_all_in_w2v(df)  # because some text return nothing, must remove ground truth too
    np_text_array = extract_numpy_vectors_from_w2v_labels(df)
    return np_text_array


def extract_numpy_vectors_from_w2v_labels(df):
    text = np.array(df[COMMENT_TEXT_INDEX].as_matrix())
    return text


def extract_truth_labels_as_dict(df):
    dictionary_of_truth_labels = {}
    for key in TRUTH_LABELS:
        value = np.array(df[key].as_matrix())
        dictionary_of_truth_labels[key] = value
    return dictionary_of_truth_labels


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime(FILE_NAME_STRING_FORMATING) + FILE_NAME_STRING_DELIMITER + expt_name
    os.makedirs(KERAS_MODEL_DIRECTORY.format(folder), exist_ok=True)
    model.save(MODEL_SAVE_PATH.format(folder))
    with open(TRAIN_HISTORY_DICT_PATH.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


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


def build_keras_model():
    from keras.models import Sequential
    from keras.layers import Dense
    data_dim = 300
    timesteps = 50

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    from keras.layers import LSTM

    model.add(LSTM(64, return_sequences=True, input_shape=(3000, 300)))
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def build_keras_embeddings_model(max_size):
    from keras.models import Sequential
    from keras.layers import Dense

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    from keras.layers import LSTM, Embedding

    model.add(Embedding(max_size, 64, input_length=3000))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def split_train_test(np_text_array, truth_dictionary):
    from sklearn.model_selection import train_test_split
    data_dictionary = {}
    for key in truth_dictionary:
        X_train, X_test, y_train, y_test = train_test_split(np_text_array, truth_dictionary[key], test_size=0.1,
                                                            random_state=42)

        data_dictionary[key] = {X_TRAIN_DATA_INDEX: X_train, X_TEST_DATA_INDEX: X_test, Y_TRAIN_DATA_INDEX: y_train,
                                Y_TEST_DATA_INDEX: y_test}
    return data_dictionary


def drop_words_with_no_vectors_at_all_in_w2v(df):
    df.drop(df[df.comment_text.isnull()].index, inplace=True)


if __name__ == "__main__":
    """
    EXPT_NAME = "TEST_W2V"
    USE_WORD2VEC = False
    SAMPLE_DATA_FILE = './data/sample.csv'
    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    print(EXPT_NAME)
    main(SAMPLE_DATA_FILE, model, testing=True, expt_name=EXPT_NAME)

    EXPT_NAME = "TEST_NONW2V"
    USE_WORD2VEC = True
    SAMPLE_DATA_FILE = './data/sample.csv'
    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    print(EXPT_NAME)
    main(SAMPLE_DATA_FILE, model, testing=True, expt_name=EXPT_NAME)
    """
    print("done with tests, loading true model")
    USE_WORD2VEC = False
    EXPT_NAME = "REAL"
    DATA_FILE = './data/train.csv'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    model = load_w2v_model_from_path(W2V_MODEL, binary_input=True)
    main(DATA_FILE, model, testing=False, expt_name=EXPT_NAME)
