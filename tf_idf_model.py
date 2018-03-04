from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from scipy import sparse

from utils import COMMENT_TEXT_INDEX, TRUTH_LABELS, load_data, dataframe_to_list


def tf_idf_vectorisor(df):
    """
    function should return tf-idf logistic regression score
    :param df: list
    :type df: string
    :return: sparse matrix
    :rtype: value
    """
    vect_char = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(3, 6))
    vect_char2 = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(2, 5))
    vect_word = TfidfVectorizer(stop_words='english')
    lst = dataframe_to_list(df)
    sparse_matrix_word = vect_word.fit_transform(lst)
    sparse_matrix_char = vect_char.fit_transform(lst)
    sparse_matrix_char2 = vect_char2.fit_transform(lst)
    sparse_matrix_combined = sparse.hstack([sparse_matrix_word, sparse_matrix_char, sparse_matrix_char2])
    #print("\nFeatures of vectorizer_character\n", vect_char.get_feature_names())
    #print("\nRemoved Features of vectorizer_character \n", vect_char.get_stop_words())
    #print("\nHyperparameters of vectorizer_character\n", vect_char.fit(lst))
    #print("\nFeatures of vectorizer_word\n", vect_word.get_feature_names())
    #print("\nRemoved Features of vectorizer_word \n", vect_word.get_stop_words())
    #print("\nHyperparameters of vectorizer_word\n", vect_word.fit(lst))
    return sparse_matrix_combined
   
def build_logistic_regression_model(vector,df):
    y = df[TRUTH_LABELS]
    log_dict = {}
    for i, col in enumerate(TRUTH_LABELS):
        lr = LogisticRegression(random_state=i, class_weight=None, solver='saga', n_jobs=-1, multi_class='multinomial')
        print("Building {} model for column:{""}".format(i, col))
        lr.fit(vector, y[col])
        pred = lr.predict(vector)
        col = str(col)
        print("Column:", col)
        print('\nConfusion matrix\n', confusion_matrix(y[col], pred))
        print(classification_report(y[col], pred))
        log_dict[lr] = str(i)
    return log_dict


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(DATA_FILE)
    vector = tf_idf_vectorisor(df[COMMENT_TEXT_INDEX])
    aggressively_positive_model_report = build_logistic_regression_model(vector, df)