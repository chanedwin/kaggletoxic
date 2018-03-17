import lda.datasets
import re

from sklearn.feature_extraction.text import CountVectorizer
from utils import initalise_logging,load_data, COMMENT_TEXT_INDEX,extract_truth_labels_as_dict
from tf_idf_model import build_logistic_regression_model


def get_lda_topics(sentences):
    search_and_replace_numerals_with_space = lambda x: re.sub(r'(\d[\d\.])+', '', x.lower())
    vectorizer = CountVectorizer(preprocessor=search_and_replace_numerals_with_space, stop_words='english', min_df=20)
    sentences = sentences.tolist()
    vectorizer.fit(sentences)
    logger
    logger.info("feature_words %s", vectorizer.get_feature_names())
    tf_idf_sparse_matrix = vectorizer.transform(sentences)
    model = lda.LDA(n_topics=2000, n_iter=5, random_state=1)
    topics = model.fit_transform(tf_idf_sparse_matrix)
    return model, topics


def predict_lda_topics(model, sentences):
    search_and_replace_numerals_with_space = lambda x: re.sub(r'(\d[\d\.])+', '', x.lower())
    vectorizer = CountVectorizer(preprocessor=search_and_replace_numerals_with_space, stop_words='english', min_df=20)
    vectorizer.fit(sentences)
    logger.info("feature_words %s", vectorizer.get_feature_names())
    tf_idf_sparse_matrix = vectorizer.transform(sentences)
    topics = model.transform(tf_idf_sparse_matrix)
    return topics


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/balanced_train_file.csv'
    logger = initalise_logging('./data/Log_files/')
    train_df = load_data(DATA_FILE)
    truth_dictionary = extract_truth_labels_as_dict(train_df)
    train_sentences = train_df[COMMENT_TEXT_INDEX]
    model, topics = get_lda_topics(train_sentences)
    predict = predict_lda_topics(model, train_sentences)
    lr = build_logistic_regression_model(predict, truth_dictionary, logger=logger)