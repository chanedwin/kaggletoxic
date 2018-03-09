import os
import pickle
import time

import keras.callbacks
from keras import backend as K
from keras.layers import Dense
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

from utils import TRUTH_LABELS, COMMENT_TEXT_INDEX
from utils import transform_text_in_df_return_w2v_np_vectors, split_train_test
from sklearn.metrics import confusion_matrix, classification_report

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

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


def lstm_main(summarized_sentences,predicted_data, truth_dictionary, w2v_model, testing, use_w2v=True):
    if testing:
        print("running tests")
        number_of_epochs = 1
    else:
        print("running eval")
        number_of_epochs = 1

    prediction_sentences = predicted_data[COMMENT_TEXT_INDEX]

    # process data
    print("processing data")
    if use_w2v:
        np_text_array = transform_text_in_df_return_w2v_np_vectors(summarized_sentences, w2v_model)
        np_predict_array = transform_text_in_df_return_w2v_np_vectors(prediction_sentences, w2v_model)
        data_dict = split_train_test(np_text_array, truth_dictionary)
        model_dict = {}
        history_dict = {}
        results_dict = {}
        for key in TRUTH_LABELS:
            x_train = data_dict[key][X_TRAIN_DATA_INDEX]
            padded_x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
            y_train = data_dict[key][Y_TRAIN_DATA_INDEX]

            x_test = data_dict[key][X_TEST_DATA_INDEX]
            padded_x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
            y_test = data_dict[key][Y_TEST_DATA_INDEX]

            model = build_keras_model()
            print("training network")
            early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

            history = model.fit(padded_x_train, y_train,
                                batch_size=1024, epochs=number_of_epochs,
                                validation_data=(padded_x_test, y_test),
                                callbacks=[early_stop_callback, ]
                                )
            validation = model.predict_classes(padded_x_test)
            print(y_test, validation)

            print('\nConfusion matrix\n', confusion_matrix(y_test, validation))
            print(classification_report(y_test, validation))
            padded_x_predict_train = sequence.pad_sequences(np_predict_array, maxlen=MAXLEN)
            results = model.predict(padded_x_predict_train)
            history_dict[key] = history.history
            model_dict[key] = model
            results_dict[key] = results
            # try some values
        return model_dict, history_dict, results_dict
    else:

        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_ONE_HOT,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=" ",
                              char_level=False)
        tokenizer.fit_on_texts(summarized_sentences)
        transformed_text = tokenizer.texts_to_sequences(summarized_sentences)
        predicted_transformed_text = tokenizer.texts_to_sequences(prediction_sentences)
        vocab_size = len(tokenizer.word_counts)

        print("vocab length is", len(tokenizer.word_counts))

        data_dict = split_train_test(transformed_text, truth_dictionary)
        model_dict = {}
        history_dict = {}
        results_dict = {}
        for key in TRUTH_LABELS:
            x_train = data_dict[key][X_TRAIN_DATA_INDEX]
            padded_x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
            y_train = data_dict[key][Y_TRAIN_DATA_INDEX]

            x_test = data_dict[key][X_TEST_DATA_INDEX]
            padded_x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
            y_test = data_dict[key][Y_TEST_DATA_INDEX]

            # build neural network model
            print("training network")
            early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')
            model = build_keras_embeddings_model(vocab_size)

            history = model.fit(padded_x_train, y_train,
                                batch_size=MAX_BATCH_SIZE_PRE_TRAINED, epochs=number_of_epochs,
                                validation_data=(padded_x_test, y_test),
                                callbacks=[early_stop_callback, ]
                                )
            validation = model.predict_classes(padded_x_test)
            print('\nConfusion matrix\n', confusion_matrix(y_test, validation))
            print(classification_report(y_test, validation))
            padded_predicted_transformed_text = sequence.pad_sequences(predicted_transformed_text, maxlen=MAXLEN)
            results = model.predict_classes(padded_predicted_transformed_text)
            history_dict[key] = history.history
            model_dict[key] = model
            results_dict[key] = results
        return model_dict, history_dict,results_dict  # THIS IS FAKE


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime(FILE_NAME_STRING_FORMATING) + FILE_NAME_STRING_DELIMITER + expt_name
    os.makedirs(KERAS_MODEL_DIRECTORY.format(folder), exist_ok=True)
    model.save(MODEL_SAVE_PATH.format(folder))
    with open(TRAIN_HISTORY_DICT_PATH.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def build_keras_model(testing = False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(3000, 300)))
    if not testing :
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def build_keras_embeddings_model(max_size,testing = False):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Embedding(max_size, 64, input_length=3000))
    if not testing :
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

