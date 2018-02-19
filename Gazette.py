from utils import load_data, dataframe_to_list, COMMENT_TEXT_INDEX
from keras.preprocessing.text import Tokenizer
























if __name__ == "__main__":
    UNPROCESSED_BAD_WORDS = './data/Bad_words/google_twunter_lol'
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(SAMPLE_DATA_FILE)
    data_text = dataframe_to_list(df)
    print(data_text)

