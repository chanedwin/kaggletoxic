import os
import pickle
import time

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn.model_selection import train_test_split

from gazette_model import process_bad_words
from lda_model import get_lda_topics
from lsi_model import build_LSI_model
from lstm_model import lstm_main, lstm_predict, MAX_VOCAB_SIZE
from tf_idf_model import tf_idf_vectorizer_small, tf_idf_vectorizer_big, build_logistic_regression_model
from utils import COMMENT_TEXT_INDEX, TRUTH_LABELS, BALANCED_DATA_FILE

IGNORE_FLAG = 0
TRAIN_NEW_FLAG = 1
USE_OLD_FLAG = 2
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
    predict_df = load_data(predict_data_file)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(predict_df, pd.DataFrame)

    # get truth dictionary
    truth_dictionary = extract_truth_labels_as_dict(train_df)
    train_sentences = train_df[COMMENT_TEXT_INDEX]
    summarized_sentences = summarized_sentences[:len(train_df)]

    if testing:
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()
        truth_dictionary.popitem()

    # get gazette matrices
    if train_flag_dict[GAZETTE_FLAG]:
        if train_new:
            sparse_gazette_matrices = process_bad_words(train_sentences)
            np.save(save_file_directory + SPARSE_ARRAY_NAME, sparse_gazette_matrices)
            assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
        else:
            sparse_gazette_matrices = np.load(save_file_directory + SPARSE_ARRAY_NAME)
            assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
            logger.info("done getting sparse matrices of shape %s", sparse_gazette_matrices.shape)
            del sparse_gazette_matrices

    # get w2v lstm matrices
    if train_flag_dict[W2V_FLAG]:
        if train_new:
            w2v_model_dict, w2v_result_dict = lstm_main(summarized_sentences=summarized_sentences,
                                                        truth_dictionary=truth_dictionary,
                                                        w2v_model=w2v_model, testing=testing,
                                                        use_w2v=True, logger=logger)
            for model_name in w2v_model_dict:
                model = w2v_model_dict[model_name]
                model.save(save_file_directory + model_name + PRE_TRAINED_RESULT)
        else:
            w2v_model_dict = {}
            for key in TRUTH_LABELS:
                w2v_model_dict[key] = load_model(save_file_directory + key + PRE_TRAINED_RESULT)
        w2v_results = lstm_predict(model_dict=w2v_model_dict, tokenizer=None, predicted_data=train_df,
                                   truth_dictionary=truth_dictionary,
                                   w2v_model=w2v_model,
                                   use_w2v=True, logger=logger)
        logger.info(w2v_results)

    # get novel lstm matrices
    if train_flag_dict[NOVEL_FLAG]:
        if train_new:
            novel_model_dict, w2v_result_dict, tokenizer = lstm_main(
                summarized_sentences=summarized_sentences,
                truth_dictionary=truth_dictionary,
                w2v_model=None, testing=testing,
                use_w2v=False, logger=logger)
            for model_name in novel_model_dict:
                model = novel_model_dict[model_name]
                model.save(save_file_directory + model_name + NOVEL_TRAINED_RESULT)
        else:
            novel_model_dict = {}
            from keras.preprocessing.text import Tokenizer

            tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE,
                                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                  lower=True,
                                  split=" ",
                                  char_level=False)
            tokenizer.fit_on_texts(summarized_sentences)
            for key in TRUTH_LABELS:
                novel_model_dict[key] = load_model(save_file_directory + key + NOVEL_TRAINED_RESULT)
        novel_results = lstm_predict(model_dict=novel_model_dict, predicted_data=train_df, tokenizer=tokenizer,
                                     truth_dictionary=truth_dictionary,
                                     w2v_model=None,
                                     use_w2v=False, logger=logger)
        logger.info(novel_results)

    # get tf-idf vectorizer
    if train_flag_dict[TF_IDF_FLAG]:
        if train_new:
            vector_small = tf_idf_vectorizer_small(train_sentences, logger=logger)
            logger.info(vector_small)
            np.save(save_file_directory + TF_IDF_SMALL, vector_small)
        else:
            vector_small = np.load(save_file_directory + TF_IDF_SMALL)
        # get log regression score from tf-idf(2-6 n gram) log reg
        if train_new:
            vector_big = tf_idf_vectorizer_big(train_sentences, logger=logger)
            logger.info(vector_big)
            aggressively_positive_model_report = build_logistic_regression_model(vector_big, truth_dictionary,
                                                                                 logger=logger)
            np.save(save_file_directory + TF_IDF_BIG, vector_big)
            logger.info(aggressively_positive_model_report)
        else:
            vector_big = np.load(save_file_directory + TF_IDF_BIG)
            aggressively_positive_model_report = build_logistic_regression_model(vector_big, truth_dictionary,
                                                                                 logger=logger)
            logger.info(aggressively_positive_model_report)

    # get lsi
    if train_flag_dict[LSI_FLAG]:
        if train_new:
            topics = build_LSI_model(train_sentences)
            logger.info("topics are %s", topics)
            np.save(save_file_directory + LSI_MODEL, topics)
            lsi_topics = build_LSI_model(train_sentences)
            np.save(save_file_directory + LSI_MODEL, lsi_topics)
        else:
            np.load(save_file_directory + LSI_MODEL)

    # get lda
    if train_flag_dict[LDA_FLAG]:
        if train_new:
            topics = get_lda_topics(train_sentences)
            logger.info("topics are %s", topics)
            np.save(save_file_directory + LDA_MODEL, topics)
            lda_topics = get_lda_topics(train_sentences)
            np.save(save_file_directory + LDA_MODEL, lda_topics)
        else:
            np.load(save_file_directory + LDA_MODEL)

    "sparse = {}, w2v_lstm = {}, novel_lstm = {}, tf-idf = {}, lda = {}, lsi = {}"
    logger.info(sparse_gazette_matrices.shape)
    sparse_gazette_matrices = np.load(save_file_directory + SPARSE_ARRAY_NAME)
    assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
    for key in aggressively_positive_model_report:
        aggressively_positive_model_report[key] = np.array(
            [i[1] for i in aggressively_positive_model_report[key]]).reshape((50, 1))
    for key in truth_dictionary:
        np_full_array = np.hstack(
            (sparse_gazette_matrices, w2v_results[key], novel_results[key], lsi_topics, lda_topics,
             aggressively_positive_model_report[key]))
        logger.info("shape of array for wide network is", np_full_array.shape)
        deep_and_wide_network(np_full_array=np_full_array,
                              testing=testing,
                              truth_dictionary=truth_dictionary, key=key)


def deep_and_wide_network(np_full_array, testing, truth_dictionary, key):
    # get w2v lstm matrices
    if testing:
        number_of_epochs = 1
    else:
        number_of_epochs = 5

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
    sparse_model.fit(full_x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1)
    sparse_model.evaluate(full_x_test, y_test)
    return sparse_model


if __name__ == "__main__":
    SUM_SENTENCES_FILE = './data/newtrain.p'
    summarized_sentence_data = pickle.load(open(SUM_SENTENCES_FILE, "rb"))

    SAMPLE_DATA_FILE = './data/sample.csv'
    TRAIN_DATA_FILE = './data/small_train.csv'
    PREDICT_DATA_FILE = './data/test_predict.csv'

    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    sample_model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)

    # -----------------------------------------------------------------------------------------------------------------
    # SUPER IMPORTANT FLAG

    train_new = True  # True if training new model, else false
    EXPT_NAME = "09_03_18_10_00_49"  # ONLY USED OF train_new = False
    feature_dictionary = {GAZETTE_FLAG: TRAIN_NEW_FLAG,
                          W2V_FLAG: TRAIN_NEW_FLAG,
                          NOVEL_FLAG: TRAIN_NEW_FLAG,
                          LDA_FLAG: TRAIN_NEW_FLAG,
                          LSI_FLAG: TRAIN_NEW_FLAG,
                          TF_IDF_FLAG: TRAIN_NEW_FLAG,
                          FAST_TEXT_FLAG: IGNORE_FLAG}
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
        # real_model = load_w2v_model_from_path(W2V_MODEL)
        main(train_data_file=BALANCED_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=True,
             train_flag_dict=feature_dictionary, logger=real_logger)
        """
    else:
        test_logger = initalise_logging(TEST_SAVE_FILE_PATH)
        real_logger = initalise_logging(REAL_SAVE_FILE_PATH)
        logger.info("preparing to reuse old model using flags %s", feature_dictionary)
        SAVE_FILE_PATH = "./expt/" + EXPT_NAME + ""
        TEST_SAVE_FILE_PATH = SAVE_FILE_PATH + "_TEST/"
        REAL_SAVE_FILE_PATH = SAVE_FILE_PATH + "_REAL/"
        try:
            assert os.path.exists(TEST_SAVE_FILE_PATH)
            assert os.path.exists(REAL_SAVE_FILE_PATH)
        except:
            raise Exception("Experiment path doesn't exist")

        logger.info("doing tests")
        main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=True, save_file_directory=TEST_SAVE_FILE_PATH, train_new=False,
             train_flag_dict=feature_dictionary)

        ""
        logger.info("starting expt")
        real_model = load_w2v_model_from_path(W2V_MODEL)  # doing this at the end cause very slow
        main(train_data_file=TRAIN_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=real_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=False,
             train_flag_dict=feature_dictionary)
        """
