import os
import pickle
import time

import keras.callbacks
from keras import backend as K
from keras.layers import Dense
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

from utils import TOXIC_TEXT_INDEX, COMMENT_TEXT_INDEX
from utils import transform_text_in_df_return_w2v_np_vectors, extract_truth_labels_as_dict, split_train_test, load_data, \
    tokenize_tweets

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


def lstm_main(data_file, w2v_model, testing, use_w2v=True, expt_name="test"):
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
    if use_w2v:
        tokenize_tweets(df)
        np_text_array = transform_text_in_df_return_w2v_np_vectors(df, w2v_model)
        truth_dictionary = extract_truth_labels_as_dict(df)
        data_dict = split_train_test(np_text_array, truth_dictionary)
        key = TOXIC_TEXT_INDEX  # testing with 1 key for now
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

        # try some values
        x_predict = model.predict(padded_x_test[::1000])
        for index, val in enumerate(x_predict):
            print("predicted is {}, truth is {},".format(x_predict[index][0], y_train[index]))
        save_model_details_and_training_history(expt_name, history, model)
        return x_predict  # THIS IS FAKE
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
        model = build_keras_embeddings_model(vocab_size)

        history = model.fit(padded_x_train, y_train,
                            batch_size=MAX_BATCH_SIZE_PRE_TRAINED, epochs=number_of_epochs,
                            validation_data=(padded_x_test, y_test),
                            callbacks=[early_stop_callback, ]
                            )

        # try some values
        x_predict = model.predict(padded_x_test[::1000])
        for index, val in enumerate(x_predict):
            print("predicted is {}, truth is {},".format(x_predict[index][0], y_train[index]))
        save_model_details_and_training_history(expt_name, history, model)
        return x_predict  # THIS IS FAKE


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime(FILE_NAME_STRING_FORMATING) + FILE_NAME_STRING_DELIMITER + expt_name
    os.makedirs(KERAS_MODEL_DIRECTORY.format(folder), exist_ok=True)
    model.save(MODEL_SAVE_PATH.format(folder))
    with open(TRAIN_HISTORY_DICT_PATH.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def build_keras_model():
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

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
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

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


def tf_idf_summarizer(text):
    import re
    document = re.sub("[ ^A-Za-z.-]+"," ", text)

    document = document.replace("-","")
    document = document.replace("...","")
    document = document.replace("Mr.", "Mr").replace("Mrs.", "Mrs")

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(document)
    freq_term_matrix = count_vect.transform(train_data)

    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)

    doc_freq_term = count_vect.transform([doc])
    doc_tfidf_matrix = tfidf.transform(doc_freq_term)


