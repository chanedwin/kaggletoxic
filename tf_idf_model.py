from utils import TOXIC_TEXT_INDEX, load_data


def return_tf_idf_score(df):
    """
    function should return tf-idf logistic regression score
    :param df:
    :type df:
    :return:
    :rtype:
    """
    print(df)


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'

    df = load_data(SAMPLE_DATA_FILE)
    print(df[TOXIC_TEXT_INDEX])
