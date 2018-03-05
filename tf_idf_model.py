from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from scipy import sparse
from utils import COMMENT_TEXT_INDEX, TRUTH_LABELS, load_data, dataframe_to_list


def tf_idf_vectorisor_word(lst):

    """
    function should return tf-idf logistic regression score
    :param : list
    :type : string
    :return: sparse matrix
    :rtype: value
    """
    vect_word = TfidfVectorizer(stop_words='english')
    sparse_matrix_word = vect_word.fit_transform(lst)
    print("\nshape of word vector\n", sparse_matrix_word.shape)
    #print("\nFeatures of vectorizer_character\n", vect_char.get_feature_names())
    #print("\nRemoved Features of vectorizer_character \n", vect_char.get_stop_words())
    #print("\nHyperparameters of vectorizer_character\n", vect_char.fit(lst))
    #print("\nFeatures of vectorizer_word\n", vect_word.get_feature_names())
    #print("\nRemoved Features of vectorizer_word \n", vect_word.get_stop_words())
    #print("\nHyperparameters of vectorizer_word\n", vect_word.fit(lst))
    return sparse_matrix_word


def tf_idf_vectorisor_big(lst):
    """
    function should return tf-idf logistic regression score
    :param : list
    :type : string
    :return: sparse matrix
    :rtype: value
    """
    vect_char = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(2, 6))
    vect_word = TfidfVectorizer(stop_words='english')
    sparse_matrix_word = vect_word.fit_transform(lst)
    sparse_matrix_char = vect_char.fit_transform(lst)
    sparse_matrix_combined = sparse.hstack([sparse_matrix_word, sparse_matrix_char])
    print("\nshape of Big vector\n", sparse_matrix_combined.shape)
    #print("\nFeatures of vectorizer_character\n", vect_char.get_feature_names())
    #print("\nRemoved Features of vectorizer_character \n", vect_char.get_stop_words())
    #print("\nHyperparameters of vectorizer_character\n", vect_char.fit(lst))
    #print("\nFeatures of vectorizer_word\n", vect_word.get_feature_names())
    #print("\nRemoved Features of vectorizer_word \n", vect_word.get_stop_words())
    #print("\nHyperparameters of vectorizer_word\n", vect_word.fit(lst))
    return sparse_matrix_combined

def tf_idf_vectorisor_small(lst):
    """
    function should return tf-idf logistic regression score
    :param : list
    :type : string
    :return: sparse matrix
    :rtype: value
    """
    vect_char = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(3, 3))
    vect_word = TfidfVectorizer(stop_words='english')
    sparse_matrix_word = vect_word.fit_transform(lst)
    sparse_matrix_char = vect_char.fit_transform(lst)
    sparse_matrix_combined = sparse.hstack([sparse_matrix_word, sparse_matrix_char])
    print("\nshape of Small vector\n", sparse_matrix_combined.shape)
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
    dict_of_pred_probability = {}
    for i, col in enumerate(TRUTH_LABELS):
        lr = LogisticRegression(random_state=i, class_weight=None, solver='saga',n_jobs=-1, multi_class='multinomial')
        print("Building {} model for column:{""}".format(i, col))
        lr.fit(vector, y[col])
        pred = lr.predict(vector)
        col = str(col)
        print("Column:", col)
        print('\nConfusion matrix\n', confusion_matrix(y[col], pred))
        print(classification_report(y[col], pred))
        log_dict[lr] = str(col)
        dict_of_pred_probability[str(col)] = dict(lr.predict_log_proba(vector))
    return dict_of_pred_probability     


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(DATA_FILE)
    lst = dataframe_to_list(df[COMMENT_TEXT_INDEX])
    lst = [w.replace("", '') for w in lst]
    vector_big = tf_idf_vectorisor_big(lst)
    #vector_small = tf_idf_vectorisor_small(df[COMMENT_TEXT_INDEX])
    aggressively_positive_model = build_logistic_regression_model(vector_big, df)
    #print(aggressively_positive_model)
