from utils import load_data, dataframe_to_list, concatnator, COMMENT_TEXT_INDEX
import pandas as pd


def bad_word_processor(data):

    # pd.set_option('display.max_rows', 4400)
    df = pd.read_csv(data, sep='delimiter', header=None)
    print(df)
    return df




if __name__ == "__main__":
    UNPROCESSED_BAD_WORDS_DATA = './data/bad_words'
    ARABIC_PERSIAN_BAD_WORDS = './data/arabic_persian'
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(SAMPLE_DATA_FILE)
    bad_words = bad_word_processor(UNPROCESSED_BAD_WORDS_DATA)
    ap_bad_words = bad_word_processor(ARABIC_PERSIAN_BAD_WORDS)
    print(ap_bad_words)
    print(bad_words)

