from utils import TOXIC_TEXT_INDEX, load_data


def return_tf_idf_score(df):
    """
    function should return tf-idf logistic regression score
    :param df:
    :type df:
    :return:
    :rtype:
    """
    score = 0.9
    return score


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(SAMPLE_DATA_FILE)
    print(return_tf_idf_score(df))
    df = load_data(DATA_FILE)
    print(return_tf_idf_score(df))
