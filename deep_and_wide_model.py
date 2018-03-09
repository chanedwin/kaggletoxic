import os
import pickle
import time

import numpy as np
import pandas as pd
from keras import Sequential
from keras.models import load_model

from gazette_model import process_bad_words
from lda_model import get_lda_topics
from lsi_model import build_LSI_model
from lstm_model import lstm_main, lstm_predict, MAX_NUM_WORDS_ONE_HOT
from tf_idf_model import tf_idf_vectorizer_small, tf_idf_vectorizer_big, build_logistic_regression_model
from utils import COMMENT_TEXT_INDEX, TRUTH_LABELS
from utils import load_w2v_model_from_path, load_data, extract_truth_labels_as_dict

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


def main(train_data_file, predict_data_file, summarized_sentences, w2v_model, testing, save_file_directory="",
         train_new=True, train_flag_dict=None):
    assert type(summarized_sentences) == list
    assert type(summarized_sentences[0]) == str
    train_df = load_data(train_data_file)
    predict_df = load_data(predict_data_file)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(predict_df, pd.DataFrame)

    if testing:
        summarized_sentences = summarized_sentences[:len(train_df)]

    # get truth dictionary
    truth_dictionary = extract_truth_labels_as_dict(train_df)
    train_sentences = train_df[COMMENT_TEXT_INDEX]

    # get gazette matrices
    if train_flag_dict[GAZETTE_FLAG]:
        if train_new:
            sparse_gazette_matrices = process_bad_words(train_sentences)
            np.save(save_file_directory + SPARSE_ARRAY_NAME, sparse_gazette_matrices)
            assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
        else:
            sparse_gazette_matrices = np.load(save_file_directory + SPARSE_ARRAY_NAME)
            assert sparse_gazette_matrices.shape == (len(train_sentences), 3933)
        print("done getting sparse matrices of shape", sparse_gazette_matrices.shape)

    # get w2v lstm matrices
    if train_flag_dict[W2V_FLAG]:
        if train_new:
            w2v_model_dict, w2v_history_dict, w2v_result_dict = lstm_main(summarized_sentences=summarized_sentences,

                                                                          truth_dictionary=truth_dictionary,
                                                                          w2v_model=w2v_model, testing=testing,
                                                                          use_w2v=True)
            for model_name in w2v_model_dict:
                model = w2v_model_dict[model_name]
                model.save(save_file_directory + model_name + PRE_TRAINED_RESULT)
        else:
            w2v_model_dict = {}
            for key in TRUTH_LABELS:
                w2v_model_dict[key] = load_model(save_file_directory + key + PRE_TRAINED_RESULT)
        w2v_results = lstm_predict(model_dict=w2v_model_dict, tokenizer=None, predicted_data=predict_df,
                                   truth_dictionary=truth_dictionary,
                                   w2v_model=w2v_model,
                                   use_w2v=True)
        print(w2v_results)

    # get novel lstm matrices
    if train_flag_dict[NOVEL_FLAG]:
        if train_new:
            novel_model_dict, novel_history_dict, w2v_result_dict, tokenizer = lstm_main(
                summarized_sentences=summarized_sentences,
                truth_dictionary=truth_dictionary,
                w2v_model=None, testing=testing,
                use_w2v=False)
            for model_name in novel_model_dict:
                model = novel_model_dict[model_name]
                model.save(save_file_directory + model_name + NOVEL_TRAINED_RESULT)
        else:
            novel_model_dict = {}
            from keras.preprocessing.text import Tokenizer

            tokenizer = Tokenizer(num_words=MAX_NUM_WORDS_ONE_HOT,
                                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                  lower=True,
                                  split=" ",
                                  char_level=False)
            tokenizer.fit_on_texts(summarized_sentences)
            for key in TRUTH_LABELS:
                novel_model_dict[key] = load_model(save_file_directory + key + NOVEL_TRAINED_RESULT)
        novel_results = lstm_predict(model_dict=novel_model_dict, predicted_data=predict_df, tokenizer=tokenizer,
                                     truth_dictionary=truth_dictionary,
                                     w2v_model=None,
                                     use_w2v=False)
        print(novel_results)

    # get tf-idf vectorizer
    if train_flag_dict[TF_IDF_FLAG]:
        if train_new:
            vector_small = tf_idf_vectorizer_small(train_sentences)
            print(vector_small)
            np.save(save_file_directory + TF_IDF_SMALL, vector_small)
        else:
            vector_small = np.load(save_file_directory + TF_IDF_SMALL)
        # get log regression score from tf-idf(2-6 n gram) log reg
        if train_new:
            vector_big = tf_idf_vectorizer_big(train_sentences)
            print(vector_big)
            aggressively_positive_model_report = build_logistic_regression_model(vector_big, truth_dictionary)
            np.save(save_file_directory + TF_IDF_BIG, vector_big)
            print(aggressively_positive_model_report)
        else:
            vector_big = np.load(save_file_directory + TF_IDF_BIG)
            aggressively_positive_model_report = build_logistic_regression_model(vector_big, truth_dictionary)
            print(aggressively_positive_model_report)

    # get lsi
    if train_flag_dict[LSI_FLAG]:
        if train_new:
            topics = build_LSI_model(train_sentences)
            print("topics are ", topics)
            np.save(save_file_directory + LSI_MODEL, topics)
        else:
            np.load(save_file_directory + LSI_MODEL)

    # get lda
    if train_flag_dict[LDA_FLAG]:
        if train_new:
            topics = get_lda_topics(train_sentences)
            print("topics are ", topics)
            np.save(save_file_directory + LDA_MODEL, topics)
        else:
            np.load(save_file_directory + LDA_MODEL)

    "sparse = {}, w2v_lstm = {}, novel_lstm = {}, tf-idf = {}, lda = {}, lsi = {}"
    print(sparse_gazette_matrices.shape)
    deep_and_wide_model = Sequential()


if __name__ == "__main__":
    SUM_SENTENCES_FILE = './data/newtrain.p'
    summarized_sentence_data = pickle.load(open(SUM_SENTENCES_FILE, "rb"))

    SAMPLE_DATA_FILE = './data/sample.csv'
    TRAIN_DATA_FILE = './data/train.csv'
    PREDICT_DATA_FILE = './data/test_predict.csv'

    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    sample_model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)

    # -----------------------------------------------------------------------------------------------------------------
    # SUPER IMPORTANT FLAG

    train_new = True  # True if training new model, else false
    EXPT_NAME = "09_03_18_10_00_49"  # ONLY USED OF train_new = False
    feature_dictionary = {GAZETTE_FLAG: 1,
                          W2V_FLAG: 1,
                          NOVEL_FLAG: 1,
                          LDA_FLAG: 1,
                          LSI_FLAG: 1,
                          TF_IDF_FLAG: 1,
                          FAST_TEXT_FLAG: 1}
    # -----------------------------------------------------------------------------------------------------------------

    if train_new:
        EXPT_NAME = time.strftime(FILE_NAME_STRING_FORMATING)
        SAVE_FILE_PATH = "./expt/" + EXPT_NAME + ""
        TEST_SAVE_FILE_PATH = SAVE_FILE_PATH + "_TEST/"
        REAL_SAVE_FILE_PATH = SAVE_FILE_PATH + "_REAL/"
        os.makedirs(TEST_SAVE_FILE_PATH)
        os.makedirs(REAL_SAVE_FILE_PATH)

        print("preparing to train new model")

        print("doing tests")
        main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=True, save_file_directory=TEST_SAVE_FILE_PATH, train_new=True,
             train_flag_dict=feature_dictionary)

        """
        print("starting real training")
        real_model = load_w2v_model_from_path(W2V_MODEL)
        main(train_data_file=TRAIN_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=real_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=True,
             train_flag_dict=feature_dictionary)
        """
    else:
        print("preparing to reuse old model using flags", feature_dictionary)
        SAVE_FILE_PATH = "./expt/" + EXPT_NAME + ""
        TEST_SAVE_FILE_PATH = SAVE_FILE_PATH + "_TEST/"
        REAL_SAVE_FILE_PATH = SAVE_FILE_PATH + "_REAL/"
        try:
            assert os.path.exists(TEST_SAVE_FILE_PATH)
            assert os.path.exists(REAL_SAVE_FILE_PATH)
        except:
            raise Exception("Experiment path doesn't exist")

        print("doing tests")
        main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=sample_model, testing=True, save_file_directory=TEST_SAVE_FILE_PATH, train_new=False,
             train_flag_dict=feature_dictionary)

        """
        print("starting expt")
        real_model = load_w2v_model_from_path(W2V_MODEL)  # doing this at the end cause very slow
        main(train_data_file=TRAIN_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
             summarized_sentences=summarized_sentence_data,
             w2v_model=real_model, testing=False, save_file_directory=REAL_SAVE_FILE_PATH, train_new=False,
             train_flag_dict=feature_dictionary)
        """
