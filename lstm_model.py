import os
import pickle
import time

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM, Embedding
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from utils import COMMENT_TEXT_INDEX
from utils import split_train_test
from utils import transform_text_in_df_return_w2v_np_vectors

BATCH_SIZE = 5

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

MAX_VOCAB_SIZE = 50000
MAX_NUM_WORDS_ONE_HOT = 300

FILE_NAME_STRING_DELIMITER = "_"
FILE_NAME_STRING_FORMATING = "%d_%m_%y_%H:%M"
KERAS_MODEL_DIRECTORY = 'keras_models/{}'
TRAIN_HISTORY_DICT_PATH = 'keras_models/{}/trainHistoryDict'
MODEL_SAVE_PATH = 'keras_models/{}/keras_model.h5'

MAX_W2V_LENGTH = 300


def lstm_main(summarized_sentences, truth_dictionary, w2v_model, testing, use_w2v=True, logger=None):
    if testing:
        logger.info("running tests")
        number_of_epochs = 1
    else:
        logger.info("running eval")
        number_of_epochs = 1

    # process data
    logger.info("processing data")
    if use_w2v:
        np_vector_array = transform_text_in_df_return_w2v_np_vectors(summarized_sentences, w2v_model)
        model_dict = {}
        results_dict = {}

        for key in truth_dictionary:
            x_train, x_test, y_train, y_test = train_test_split(np_vector_array, truth_dictionary[key],
                                                                test_size=0.1,
                                                                random_state=42)
            padded_x_test = sequence.pad_sequences(x_test, maxlen=MAX_W2V_LENGTH)

            model = build_keras_model(max_len=MAX_W2V_LENGTH)
            logger.info("training network")

            for e in range(number_of_epochs):
                print("epoch %d" % e)
                for X_train, Y_train in w2v_batch_generator(x_train, y_train):
                    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=1)

            validation = model.predict_classes(padded_x_test)
            logger.info('\nConfusion matrix\n %s', confusion_matrix(y_test, validation))
            logger.info('print classification report\n %s', classification_report(y_test, validation))

            model_dict[key] = model
            results_dict[key] = validation
            # try some values
        return model_dict, results_dict
    else:
        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=" ",
                              char_level=False)
        tokenizer.fit_on_texts(summarized_sentences)
        transformed_text = tokenizer.texts_to_sequences(summarized_sentences)
        for index, text in enumerate(transformed_text):
            transformed_text[index] = np.array(text, dtype='int8')
        transformed_text = np.array(transformed_text)
        print("shape of text is", transformed_text.shape)
        vocab_size = len(tokenizer.word_counts)
        logger.info("vocab length is %s", len(tokenizer.word_counts))

        data_dict = split_train_test(transformed_text, truth_dictionary)
        logger.info("vocab length is %s", len(tokenizer.word_counts))
        model_dict = {}
        results_dict = {}
        for key in truth_dictionary:
            x_train, x_test, y_train, y_test = train_test_split(transformed_text, truth_dictionary[key],
                                                                test_size=0.1,
                                                                random_state=42)
            padded_x_test = sequence.pad_sequences(x_test, maxlen=MAX_W2V_LENGTH)

            print("training network")
            model = build_keras_embeddings_model(max_vocab_size=vocab_size, max_length=MAX_NUM_WORDS_ONE_HOT)
            print("vocab size is", vocab_size)
            for e in range(number_of_epochs):
                print("epoch %d" % e)
                for X_train, Y_train in novel_batch_generator(x_train, y_train):
                    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=1)

            validation = model.predict_classes(padded_x_test)
            print('\nConfusion matrix\n', confusion_matrix(y_test, validation))
            print(classification_report(y_test, validation))
            model_dict[key] = model
            results_dict[key] = validation
        return model_dict, results_dict, tokenizer  # THIS IS FAKE


def lstm_predict(model_dict, tokenizer, predicted_data, truth_dictionary, w2v_model, use_w2v=True, logger=None):
    if use_w2v:
        prediction_sentences = predicted_data[COMMENT_TEXT_INDEX]
        np_text_array = transform_text_in_df_return_w2v_np_vectors(prediction_sentences, w2v_model)
        padded_x_test = sequence.pad_sequences(np_text_array, maxlen=MAX_W2V_LENGTH)
        results_dict = {}
        for key in truth_dictionary:
            model = model_dict[key]
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(index=-2).output)
            intermediate_output = intermediate_layer_model.predict(padded_x_test)
            results_dict[key] = np.array(intermediate_output)
    else:
        prediction_sentences = predicted_data[COMMENT_TEXT_INDEX]
        tokenized_predictions = tokenizer.texts_to_sequences(prediction_sentences)
        padded_x_test = sequence.pad_sequences(tokenized_predictions, maxlen=MAX_NUM_WORDS_ONE_HOT)
        results_dict = {}
        for key in truth_dictionary:
            model = model_dict[key]
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(index=-2).output)
            intermediate_output = intermediate_layer_model.predict(padded_x_test)
            results_dict[key] = np.array(intermediate_output)
    return results_dict


def w2v_batch_generator(x_train, y_train):
    i = BATCH_SIZE
    while i < len(x_train) + BATCH_SIZE:
        for sample in x_train:
            print(sample.shape)
        x = sequence.pad_sequences(x_train[i - BATCH_SIZE:i], maxlen=MAX_W2V_LENGTH, dtype='float16')
        for sample in x:
            print(sample[-1], sample[-1][0])
        y = y_train[i - BATCH_SIZE:i]
        yield x, y
        i += BATCH_SIZE


def novel_batch_generator(x_train, y_train):
    i = BATCH_SIZE
    while i < len(x_train) + BATCH_SIZE:
        for sample in x_train:
            print(sample.shape)
        x = sequence.pad_sequences(x_train[i - BATCH_SIZE:i], maxlen=MAX_NUM_WORDS_ONE_HOT)
        y = y_train[i - BATCH_SIZE:i]
        print(x.shape)
        yield x, y
        i += BATCH_SIZE


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime(FILE_NAME_STRING_FORMATING) + FILE_NAME_STRING_DELIMITER + expt_name
    os.makedirs(KERAS_MODEL_DIRECTORY.format(folder), exist_ok=True)
    model.save(MODEL_SAVE_PATH.format(folder))
    with open(TRAIN_HISTORY_DICT_PATH.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def build_keras_model(max_len, testing=False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(max_len, 300)))
    if not testing:
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def build_keras_embeddings_model(max_vocab_size, max_length, testing=False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Embedding(max_vocab_size, 64, input_length=max_length))
    if not testing:
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
