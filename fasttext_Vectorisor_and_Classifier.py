import fasttext
from utils import load_data, COMMENT_TEXT_INDEX, initalise_logging
from tf_idf_model import build_logistic_regression_model
import numpy as np


def size_of_vector_and_n_features_finder(df, model):
    n_sample_size = len(df)
    print(n_sample_size)
    number_of_model_words = len(model.words)
    size_of_vector = number_of_model_words * model.dim
    logger.info("shape of array %s", size_of_vector)
    n_features = int(size_of_vector / n_sample_size)
    logger.info("number of features %s", n_features)
    return n_sample_size, n_features


def fasttext_vectoriser_skipgram(data):
    model = fasttext.skipgram(data, 'model', dim=200)
    n_sample_size, n_features = size_of_vector_and_n_features_finder(data, model)
    vector_list = []
    logger.info("number of vectorised words %s", len(model.words))
    for word in model.words:
        vector_list.append(model[word])
    myarray = np.asarray(vector_list)
    logger.info(myarray.shape)
    myarray = np.pad(myarray, (1), 'constant', constant_values=0)
    logger.info("size_of_padded_vector %s", myarray.shape)
    myarray = myarray.reshape(n_sample_size, n_features)
    logger.info(myarray.shape)
    return myarray


def fasttext_vectoriser_cbow(data):
    model = fasttext.cbow(data, 'model', dim=200)
    n_sample_size, n_features = size_of_vector_and_n_features_finder(df, model)
    vector_list = []
    for word in model.words:
        vector_list.append(model[word])
    myarray = np.asarray(vector_list)
    logger.info("previous array shape %s", myarray.shape)
    myarray = myarray.reshape(n_sample_size, n_features)
    logger.info("reshaped array %s", myarray.shape)
    return myarray


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/summarized.csv'
    df = load_data(DATA_FILE)
    logger = initalise_logging('./data/Log_files/')
    comments = df[COMMENT_TEXT_INDEX].to_csv(path='./data/comments.csv')
    logger.info(len(df))
    fasttext_vectorised = fasttext_vectoriser_skipgram(DATA_FILE)
    fasttext_vectorised_cbow = fasttext_vectoriser_cbow(DATA_FILE)
    lr = build_logistic_regression_model(fasttext_vectorised_cbow, df)
