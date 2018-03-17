from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from utils import extract_truth_labels_as_dict


from utils import COMMENT_TEXT_INDEX, load_data, initalise_logging


def tf_idf_vectorizer_big(list_of_strings, choose_to_log_data=True, log_vectorised_words=False, logger=None):
    """
    function should return tf-idf logistic regression score
    :param : list
    :type : string
    :return: sparse matrix
    :rtype: value
    """
    vect_char = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(2, 6), min_df=20)
    vect_word = TfidfVectorizer(stop_words='english', min_df=20)
    sparse_matrix_word = vect_word.fit_transform(list_of_strings)
    sparse_matrix_char = vect_char.fit_transform(list_of_strings)
    sparse_matrix_combined = sparse.hstack([sparse_matrix_word, sparse_matrix_char])
    if choose_to_log_data:
        logger.info("\nbig vector shape\n %s", sparse_matrix_combined.shape)
    if log_vectorised_words:
        logger.info("\nFeatures of vectorizer_character %s", vect_char.get_feature_names())
        logger.info("\nRemoved Features of vectorizer_character %s", vect_char.get_stop_words())
        logger.info("\nHyperparameters of vectorizer_character %s", vect_char.fit(list_of_strings))
        logger.info("\nFeatures of vectorizer_word %s", vect_word.get_feature_names())
        logger.info("\nRemoved Features of vectorizer_word %s", vect_word.get_stop_words())
        logger.info("\nHyperparameters of vectorizer_word %s", vect_word.fit(list_of_strings))
    return sparse_matrix_combined


def tf_idf_vectorizer_small(list_of_strings, choose_to_log_data=True, log_vectorised_words=False, logger=None):
    """
    function should return tf-idf logistic regression score
    :param : list
    :type : string
    :return: sparse matrix
    :rtype: value
    """

    vect_word = TfidfVectorizer(stop_words='english', min_df=20)
    sparse_matrix_word = vect_word.fit_transform(list_of_strings)
    if choose_to_log_data:
        logger.info("\nsmall vector shape %s", sparse_matrix_word.shape)
    if log_vectorised_words:
        logger.info("\nFeatures of vectorizer_word %s", vect_word.get_feature_names())
        logger.info("\nRemoved Features of vectorizer_word  %s", vect_word.get_stop_words())
        logger.info("\nHyperparameters of vectorizer_word %s", vect_word.fit(list_of_strings))
    return sparse_matrix_word


def build_logistic_regression_model(vector, truth_dictionary, choose_to_log_data=True, logger=None):
    dict_of_pred_probability = {}
    logger = initalise_logging('./data/Log_files/')
    for i, col in enumerate(truth_dictionary):
        lr = LogisticRegression(random_state=i, solver='liblinear', n_jobs=-1)
        lr.fit(vector, truth_dictionary[col])
        pred = lr.predict(vector)
        col = str(col)
        if choose_to_log_data:
            #logger.info('\nConfusion matrix for ' + col + '\n %s', confusion_matrix(truth_dictionary[col], pred))
            logger.info('\nclassification report for ' + col + '\n %s', classification_report(truth_dictionary[col], pred))
        dict_of_pred_probability[str(col)] = lr.predict_proba(vector)
    return dict_of_pred_probability


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/balanced_train_file.csv'
    logger = initalise_logging('./data/Log_files/')
    train_df = load_data(DATA_FILE)
    train_sentences = train_df[COMMENT_TEXT_INDEX]
    truth_dictionary = extract_truth_labels_as_dict(train_df)
    vector_big = tf_idf_vectorizer_big(train_sentences, logger=logger)
    #vector_small = tf_idf_vectorizer_small(train_sentences, logger=logger)
    aggressively_positive_model_report = build_logistic_regression_model(vector_big, truth_dictionary,
                                                                             logger=logger)
