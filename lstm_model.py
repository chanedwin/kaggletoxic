import os
import pickle
import time

import keras.callbacks
import numpy as np
from keras.layers import Dense
from keras.layers import GRU, Embedding
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from utils import transform_text_in_df_return_w2v_np_vectors, chunks

PATIENCE = 10

TRAINING_TIME_EPOCHS = 500

W2V_TF_BATCH_SIZE = 1000
W2V_GENERATOR_BATCH_SIZE = 10000

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

NOVEL_TF_BATCH_SIZE = 1000
MAX_VOCAB_SIZE = 200000
MAX_NUM_WORDS_ONE_HOT = 300

FILE_NAME_STRING_DELIMITER = "_"
FILE_NAME_STRING_FORMATING = "%d_%m_%y_%H:%M"
KERAS_MODEL_DIRECTORY = 'keras_models/{}'
TRAIN_HISTORY_DICT_PATH = 'keras_models/{}/trainHistoryDict'
MODEL_SAVE_PATH = 'keras_models/{}/keras_model.h5'

MAX_W2V_LENGTH = 300
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def lstm_main(summarized_sentences, truth_dictionary, w2v_model, testing, use_w2v=True, logger=None):
    if testing:
        logger.info("running tests")
        grand_number_of_epochs = 1
        number_of_epochs = 10
    else:
        logger.info("running eval")
        grand_number_of_epochs = 1
        number_of_epochs = TRAINING_TIME_EPOCHS

    # process data
    logger.info("processing data")
    if use_w2v:
        np_vector_array = transform_text_in_df_return_w2v_np_vectors(summarized_sentences, w2v_model)
        model_dict = {}
        for key in truth_dictionary:
            x_train, x_test, y_train, y_test = train_test_split(np_vector_array, truth_dictionary[key],
                                                                test_size=0.1,
                                                                random_state=42)

            model = build_keras_model(max_len=MAX_W2V_LENGTH)
            logger.info("training w2v network")
            early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0,
                                                                mode='auto')

            history = model.fit(x_train, y_train, batch_size=W2V_TF_BATCH_SIZE, epochs=number_of_epochs,
                                callbacks=[early_stop_callback, ], validation_data=(x_test, y_test))
            logger.info(str(history.history))
            logger.info('getting w2v results')
            logger.info("number of epochs completed is" + str(len(history.history['loss'])))
            validation = model.predict_classes(x_test)
            logger.info('\nConfusion matrix\n %s', confusion_matrix(y_test, validation))
            logger.info('classification report\n %s', classification_report(y_test, validation))
            model_dict[key] = model
            # try some values
        return np_vector_array, model_dict
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
            transformed_text[index] = np.array(text, dtype='int32')
        transformed_text = np.array(transformed_text)
        padded_text = []
        for chunk in chunks(transformed_text, 10):
            padded_text.extend(sequence.pad_sequences(chunk, maxlen=MAX_NUM_WORDS_ONE_HOT))
        padded_text = np.array(padded_text)

        vocab_size = len(tokenizer.word_counts)
        model_dict = {}
        for key in truth_dictionary:
            x_train, x_test, y_train, y_test = train_test_split(padded_text, truth_dictionary[key],
                                                                test_size=0.1,
                                                                random_state=42)
            logger.info("training novel network")
            model = build_keras_embeddings_model(max_vocab_size=vocab_size, max_length=MAX_NUM_WORDS_ONE_HOT)
            logger.info("vocab size is" + str(vocab_size))
            early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0,
                                                                mode='auto')

            history = model.fit(x_train, y_train, batch_size=NOVEL_TF_BATCH_SIZE, epochs=number_of_epochs,
                                callbacks=[early_stop_callback, ], validation_data=(x_test, y_test))
            logger.info(str(history.history))
            logger.info("number of epochs completed is" + str(len(history.history['loss'])))
            validation = model.predict_classes(x_test)
            logger.info('getting w2v results')
            logger.info('\nConfusion matrix\n', confusion_matrix(y_test, validation))
            logger.info("classificaiton report", classification_report(y_test, validation))
            model_dict[key] = model
        return padded_text, model_dict, tokenizer


def lstm_predict(model_dict, predicted_data, truth_dictionary, use_w2v=True, logger=None):
    if use_w2v:
        padded_x_test = predicted_data
        results_dict = {}
        for key in truth_dictionary:
            model = model_dict[key]
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(index=-2).output)
            intermediate_output = intermediate_layer_model.predict(padded_x_test)
            results_dict[key] = np.array(intermediate_output)
    else:
        padded_x_test = predicted_data
        results_dict = {}
        for key in truth_dictionary:
            model = model_dict[key]
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(index=-2).output)
            intermediate_output = intermediate_layer_model.predict(padded_x_test)
            results_dict[key] = np.array(intermediate_output, dtype='float16')
    return results_dict


def w2v_batch_generator(x_train, y_train):
    batch_size = W2V_GENERATOR_BATCH_SIZE
    i = batch_size
    while i < len(x_train) + batch_size:
        x = sequence.pad_sequences(x_train[i - batch_size:i], maxlen=MAX_W2V_LENGTH, padding='pre', truncating='pre',
                                   dtype='float16')
        y = y_train[i - batch_size:i]
        yield x, y
        i += batch_size


def novel_batch_generator(x_train, y_train):
    i = NOVEL_TF_BATCH_SIZE
    while i < len(x_train) + NOVEL_TF_BATCH_SIZE:
        x = sequence.pad_sequences(x_train[i - NOVEL_TF_BATCH_SIZE:i], maxlen=MAX_NUM_WORDS_ONE_HOT)
        y = y_train[i - NOVEL_TF_BATCH_SIZE:i]
        yield x, y
        i += NOVEL_TF_BATCH_SIZE


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime(FILE_NAME_STRING_FORMATING) + FILE_NAME_STRING_DELIMITER + expt_name
    os.makedirs(KERAS_MODEL_DIRECTORY.format(folder), exist_ok=True)
    model.save(MODEL_SAVE_PATH.format(folder))
    with open(TRAIN_HISTORY_DICT_PATH.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def build_keras_model(max_len, testing=False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(GRU(300, return_sequences=True, input_shape=(max_len, 300)))
    if not testing:
        model.add(GRU(200, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(GRU(128, return_sequences=True))
        model.add(GRU(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(GRU(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(GRU(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(GRU(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def build_keras_embeddings_model(max_vocab_size, max_length, testing=False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Embedding(max_vocab_size, 300, input_length=max_length))
    if not testing:
        model.add(GRU(200, return_sequences=True))
        model.add(GRU(128, return_sequences=True))
        model.add(GRU(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(GRU(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(GRU(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(GRU(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
