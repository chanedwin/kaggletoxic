import os
import pickle
import time

import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from gazette_model import process_bad_words
from lda_model import get_lda_topics, predict_lda_topics
from lsi_model import build_LSI_model, predict_LSI_model
from lstm_model import lstm_main, lstm_predict, MAX_VOCAB_SIZE, chunks, MAX_NUM_WORDS_ONE_HOT
from tf_idf_model import tf_idf_vectorizer_small, tf_idf_vectorizer_big, build_logistic_regression_model
from utils import COMMENT_TEXT_INDEX, BALANCED_DATA_FILE, transform_text_in_df_return_w2v_np_vectors

IGNORE = 0
TRAIN = 1
REUSE = 2
from utils import load_w2v_model_from_path, load_data, extract_truth_labels_as_dict, initalise_logging

FAST_TEXT_FLAG = "fast_text"
TF_IDF_FLAG = "tf-idf"
LSI_FLAG = "lsi"
LDA_FLAG = "lda"
NOVEL_FLAG = "lstm_novel"
W2V_FLAG = "lstm_w2v"
GAZETTE_FLAG = "gazette"

SUM_SENTENCES_FILE = './data/newtrain.p'
FILE_NAME_STRING_DELIMITER = "_"
FILE_NAME_STRING_FORMATING = "%d_%m_%y_%H_%M_%S"
KERAS_MODEL_DIRECTORY = 'keras_models/{}'
TRAIN_HISTORY_DICT_PATH = 'keras_models/{}/trainHistoryDict'
MODEL_SAVE_PATH = 'keras_models/{}/keras_model.h5'

SPARSE_ARRAY_NAME = "sparse_array.npy"
W2V_VECTOR_NAME = "w2v_vec.npy"
NOVEL_VECTOR_NAME = "novel_vec.npy"
PRE_TRAINED_RESULT = "pre_train.npy"
NOVEL_TRAINED_RESULT = "novel_train.npy"
TF_IDF_SMALL = "tf_idf_small.npy"
TF_IDF_BIG = "tf_idf_small.npy"
LSI_MODEL = "lsi.npy"
LDA_MODEL = "lda.npy"

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

BATCH_SIZE = 100


def main(train_data_file, predict_data_file, summarized_sentences, w2v_model, testing, save_file_directory="",
         train_new=True, train_flag_dict=None, logger=None):
    assert type(summarized_sentences) == list
    assert type(summarized_sentences[0]) == str
    train_df = load_data(train_data_file)
    assert isinstance(train_df, pd.DataFrame)

    # get truth dictionary
    truth_dictionary = extract_truth_labels_as_dict(train_df)

    if not testing:
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()

    train_sentences = train_df[COMMENT_TEXT_INDEX]
    summarized_sentences = summarized_sentences[:len(train_df)]

    # get w2v lstm matrices
    if train_flag_dict[W2V_FLAG] == TRAIN:
        np_vector_array, w2v_model_dict = lstm_main(summarized_sentences=summarized_sentences,
                                                    truth_dictionary=truth_dictionary,
                                                    w2v_model=w2v_model, testing=testing,
                                                    use_w2v=True, logger=logger)
        for model_name in w2v_model_dict:
            model = w2v_model_dict[model_name]
            model.save(save_file_directory + model_name + PRE_TRAINED_RESULT)
        np.save(save_file_directory + W2V_VECTOR_NAME, np_vector_array)
        del np_vector_array
        del w2v_model_dict

    # get novel lstm matrices
    if train_flag_dict[NOVEL_FLAG] == TRAIN:
        transformed_text, novel_model_dict, tokenizer = lstm_main(
            summarized_sentences=summarized_sentences,
            truth_dictionary=truth_dictionary,
            w2v_model=None, testing=testing,
            use_w2v=False, logger=logger)
        for model_name in novel_model_dict:
            model = novel_model_dict[model_name]
            model.save(save_file_directory + model_name + NOVEL_TRAINED_RESULT)
        np.save(save_file_directory + NOVEL_VECTOR_NAME, transformed_text)

    # get tf-idf vectorizer
    vector_small = tf_idf_vectorizer_small(train_sentences, logger=logger)
    logger.info("getting tf-idf small vector of resultsd of shape", vector_small.shape)
    np.save(save_file_directory + TF_IDF_SMALL, vector_small)
    vector_big, vect_char, vect_word = tf_idf_vectorizer_big(train_sentences, logger=logger)
    logger.info(vector_big)
    lr_dict, tfidf_lr_results = build_logistic_regression_model(vector_big, truth_dictionary,
                                                                logger=logger)
    np.save(save_file_directory + TF_IDF_BIG, vector_big)
    logger.info("getting tf-idf log reg results")

    # reshaping needed because only interested in class 1
    for key in tfidf_lr_results:
        tfidf_lr_results[key] = np.array(
            [i[1] for i in tfidf_lr_results[key]]).reshape(
            (len(tfidf_lr_results[key]), 1))

    # get lsi
    lsi_model, lsi_topics = build_LSI_model(train_sentences)
    np.save(save_file_directory + LSI_MODEL, lsi_topics)

    # get lda
    lda_model, lda_topics = get_lda_topics(train_sentences)
    np.save(save_file_directory + LDA_MODEL, lda_topics)

    # get gazette matrices
    if train_flag_dict[GAZETTE_FLAG] == TRAIN:
        sparse_gazette_matrices = process_bad_words(train_sentences)
        np.save(save_file_directory + SPARSE_ARRAY_NAME, sparse_gazette_matrices)
        assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
        del sparse_gazette_matrices

    sparse_gazette_matrices = np.load(save_file_directory + SPARSE_ARRAY_NAME)
    assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
    logger.info("done getting sparse matrices of shape %s", sparse_gazette_matrices.shape)

    w2v_model_dict = {}
    np_vector_array = np.load(save_file_directory + W2V_VECTOR_NAME)
    for key in truth_dictionary:
        w2v_model_dict[key] = load_model(save_file_directory + key + PRE_TRAINED_RESULT)
    w2v_results = lstm_predict(model_dict=w2v_model_dict, predicted_data=np_vector_array,
                               truth_dictionary=truth_dictionary,
                               use_w2v=True, logger=logger)
    logger.info("done getting w2v matrices of shape")

    novel_model_dict = {}
    for key in truth_dictionary:
        novel_model_dict[key] = load_model(save_file_directory + key + NOVEL_TRAINED_RESULT)
    transformed_text = np.load(save_file_directory + NOVEL_VECTOR_NAME)
    novel_results = lstm_predict(model_dict=novel_model_dict, predicted_data=transformed_text,
                                 truth_dictionary=truth_dictionary,
                                 use_w2v=False, logger=logger)
    logger.info("done getting novel matrices of shape")

    dictionary_of_wide_model = {}
    for key in truth_dictionary:
        logger.info("training wide model now")
        for array in (
                sparse_gazette_matrices, w2v_results[key], novel_results[key], vector_small, lsi_topics, lda_topics,
                tfidf_lr_results[key]):
            print(array.shape)
        dense_vector_small = vector_small.toarray()
        np_full_array = np.hstack(
            (sparse_gazette_matrices, w2v_results[key], novel_results[key], lsi_topics, lda_topics,
             tfidf_lr_results[key]))
        logger.info("shape of array for wide network is", np_full_array.shape)
        model = deep_and_wide_network(np_full_array=np_full_array,
                                      testing=testing,
                                      truth_dictionary=truth_dictionary, key=key, logger=logger)
        dictionary_of_wide_model[key] = model

    # full prediction step
    # ------------------------ PREDICTION -----------------------
    predict_df = load_data(predict_data_file)
    assert isinstance(predict_df, pd.DataFrame)
    predict_sentences = [i for i in predict_df[COMMENT_TEXT_INDEX]]
    predicted_sparse_gazette_matrices = process_bad_words(predict_df[COMMENT_TEXT_INDEX])

    for key in truth_dictionary:
        w2v_model_dict[key] = load_model(save_file_directory + key + PRE_TRAINED_RESULT)
    np_vector_array = transform_text_in_df_return_w2v_np_vectors(predict_sentences, w2v_model)

    predict_w2v_results = lstm_predict(model_dict=w2v_model_dict, predicted_data=np_vector_array,
                                       truth_dictionary=truth_dictionary,
                                       use_w2v=True, logger=logger)
    from keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    tokenizer.fit_on_texts(summarized_sentences)
    predicted_transformed_text = tokenizer.texts_to_sequences(predict_sentences)

    padded_text = []
    for chunk in chunks(predicted_transformed_text, 10):
        padded_text.extend(sequence.pad_sequences(chunk, maxlen=MAX_NUM_WORDS_ONE_HOT))
    padded_text = np.array(padded_text)

    predict_novel_results = lstm_predict(model_dict=novel_model_dict, predicted_data=padded_text,
                                         truth_dictionary=truth_dictionary,
                                         use_w2v=True, logger=logger)

    predicted_lda_topics = predict_lda_topics(lda_model, predict_sentences)
    predicted_lsi_topics = predict_LSI_model(lsi_model, predict_sentences)

    # get tf-idf vectorizer
    #      vector_small = tf_idf_vectorizer_small(train_sentences, logger=logger)
    #     logger.info("getting tf-idf small vector of resultsd of shape", vector_small.shape)
    #    np.save(save_file_directory + TF_IDF_SMALL, vector_small)
    sparse_matrix_word = vect_word.transform(predict_sentences)
    sparse_matrix_char = vect_char.transform(predict_sentences)
    from scipy import sparse
    sparse_matrix_combined = sparse.hstack([sparse_matrix_word, sparse_matrix_char])
    predicted_tfidf_lr_results = {}
    for key in truth_dictionary:
        predicted_tfidf_lr_results[key] = lr_dict[key].predict_proba(sparse_matrix_combined)

    # reshaping needed because only interested in class 1
    for key in predicted_tfidf_lr_results:
        predicted_tfidf_lr_results[key] = np.array(
            [i[1] for i in predicted_tfidf_lr_results[key]]).reshape(
            (len(predicted_tfidf_lr_results[key]), 1))

    results_list = []
    for key in truth_dictionary:
        logger.info("predicting results now")
        for array in (predicted_sparse_gazette_matrices, predict_w2v_results[key], predict_novel_results[key],
             predicted_lda_topics,
             predicted_lsi_topics, predicted_tfidf_lr_results[key]):
            print (array.shape)
        np_full_array = np.hstack(
            (predicted_sparse_gazette_matrices, predict_w2v_results[key], predict_novel_results[key],
             predicted_lda_topics,
             predicted_lsi_topics, predicted_tfidf_lr_results[key]))
        model = dictionary_of_wide_model[key]
        results = [key] + [i for i in  model.predict_classes(np_full_array)]
        results_list.append(results)

    with open(save_file_directory + "predicted_results.csv","w") as csv_file:
        import csv
        csv_writer = csv.writer(csv_file)
        for i in range(len(results_list[0])):
            row = []
            for j in range(len(results_list)):
                row.append(results_list[j][i])
            csv_writer.writerow(row)


def deep_and_wide_network(np_full_array, testing, truth_dictionary, key, logger):
    # get w2v lstm matrices
    if testing:
        number_of_epochs = 1
    else:
        number_of_epochs = 100

    full_x_train, full_x_test, y_train, y_test = train_test_split(np_full_array, truth_dictionary[key],
                                                                  test_size=0.05, random_state=42)

    sparse_model = Sequential()
    sparse_model.add(Dense(128, input_shape=(np_full_array.shape[1],)))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(100))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(50))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(10))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(1, activation='sigmoid'))
    sparse_model.compile(optimizer='rmsprop',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

    sparse_model.fit(full_x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=number_of_epochs,
                     callbacks=[early_stop_callback, ])
    sparse_model.evaluate(full_x_test, y_test)
    validation = sparse_model.predict_classes(full_x_test)
    logger.info('\nConfusion matrix\n %s', confusion_matrix(y_test, validation))
    logger.info('classification report\n %s', classification_report(y_test, validation))
    return sparse_model


if __name__ == "__main__":
    SUM_SENTENCES_FILE = './data/balanced_train.p'
    summarized_sentence_data = pickle.load(open(SUM_SENTENCES_FILE, "rb"))
    SAMPLE_DATA_FILE = './data/sample.csv'
    TRAIN_DATA_FILE = './data/small_train.csv'
    PREDICT_DATA_FILE = './data/test_predict.csv'

    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    sample_model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)

    # -----------------------------------------------------------------------------------------------------------------
    # SUPER IMPORTANT FLAG

    train_new = False  # True if training new model, else false
    EXPT_NAME = "17_03_18_14_20_04"  # ONLY USED OF train_new = False
    feature_dictionary = {GAZETTE_FLAG: REUSE,
                          W2V_FLAG: REUSE,
                          NOVEL_FLAG: REUSE}
    # -----------------------------------------------------------------------------------------------------------------

    if train_new:
        EXPT_NAME = time.strftime(FILE_NAME_STRING_FORMATING)
        SAVE_FILE_PATH = "./expt/" + EXPT_NAME + ""
        TEST_SAVE_FILE_PATH = SAVE_FILE_PATH + "_TEST/"
        REAL_SAVE_FILE_PATH = SAVE_FILE_PATH + "_REAL/"
        os.makedirs(TEST_SAVE_FILE_PATH)
        os.makedirs(REAL_SAVE_FILE_PATH)
        test_logger = initalise_logging(TEST_SAVE_FILE_PATH)
        real_logger = initalise_logging(REAL_SAVE_FILE_PATH)

        test_logger.info("preparing to train new model")

        test_logger.info("doing tests")
        main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=True, save_file_directory=TEST_SAVE_FILE_PATH, train_new=True,
             train_flag_dict=feature_dictionary, logger=test_logger)

        real_logger.info("starting real training")
        real_model = load_w2v_model_from_path(W2V_MODEL)
        main(train_data_file=BALANCED_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=real_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=True,
             train_flag_dict=feature_dictionary, logger=real_logger)
    else:
        SAVE_FILE_PATH = "./expt/" + EXPT_NAME + ""
        REAL_SAVE_FILE_PATH = SAVE_FILE_PATH + "_TEST/"
        real_logger = initalise_logging(REAL_SAVE_FILE_PATH)
        real_logger.info("preparing to reuse old model using flags %s", feature_dictionary)
        try:
            assert os.path.exists(REAL_SAVE_FILE_PATH)
        except:
            raise Exception("Experiment path doesn't exist")
        real_logger.info("doing tests")
        main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=False,
             train_flag_dict=feature_dictionary, logger=real_logger)
