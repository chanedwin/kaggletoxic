import fasttext
from utils import load_data, COMMENT_TEXT_INDEX



def fasttext_vectorisor(data):
    model = fasttext.skipgram(data, 'model')
    print(model)












if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'
    df = load_data(SAMPLE_DATA_FILE)
    comments_csv = df[COMMENT_TEXT_INDEX].to_csv()
    fasttext_vectorised = fasttext_vectorisor(comments_csv)
