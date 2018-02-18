from utils import COMMENT_TEXT_INDEX, TRUTH_LABELS, load_data, dataframe_to_list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def return_tf_idf_sparse_matrix(df):
    """
    function should return tf-idf logistic regression score
    :param df: list
    :type df: string
    :return: sparse matrix
    :rtype: value
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    lst = dataframe_to_list(df)
    tf_idf_sparse_matrix = vectorizer.fit_transform(lst)
    idf = vectorizer.idf_
    print(dict(zip(vectorizer.get_feature_names(), idf)))
    print(vectorizer.get_feature_names())
    print(vectorizer.get_stop_words())
    print(vectorizer.fit(lst))
    return tf_idf_sparse_matrix


def build_logistic_regression_model(vector):
    y = df[TRUTH_LABELS]
    for i, col in enumerate(TRUTH_LABELS):
        lr = LogisticRegression(random_state=i, class_weight='balanced', solver='sag', n_jobs=4, max_iter=1000)
        print("Building {} model for column:{""}".format(i, col))
        lr.fit(vector, y[col])
        pred = lr.predict(vector)
        col = str(col)
        print("Column:", col)
        print('\nConfusion matrix\n', confusion_matrix(y[col], pred))
        print(classification_report(y[col], pred))
    return lr


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(DATA_FILE)
    vector = return_tf_idf_sparse_matrix(df[COMMENT_TEXT_INDEX])
    print(df[TRUTH_LABELS])
    print(build_logistic_regression_model(vector))

