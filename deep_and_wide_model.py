from gazette import process_bad_words
from ldalsi import get_lda_topics
from lstm_model import lstm_main
from tf_idf_model import tf_idf_vectorizer_small, tf_idf_vectorizer_big, build_logistic_regression_model
from utils import COMMENT_TEXT_INDEX
from utils import load_w2v_model_from_path

SUM_SENTENCES_FILE = './data/newtrain.p'


def main(train_data_file, predict_data_file, summarized_sentences, w2v_model, testing, expt_name="test"):
    # get gazette matrices
    sentences = train_data_file[COMMENT_TEXT_INDEX]
    sparse_gazette_matrices = process_bad_words(train_data_file)
    print(sparse_gazette_matrices.shape)
    print(sparse_gazette_matrices)

    # get lstm matrices
    if testing:
        pre_trained_results = lstm_main(train_data_file, w2v_model, testing, expt_name=expt_name, use_w2v=True)
        untrained_results = lstm_main(train_data_file, w2v_model, testing, expt_name=expt_name, use_w2v=False)
    else:
        pre_trained_results = lstm_main(train_data_file, w2v_model, testing=False, expt_name=EXPT_NAME, use_w2v=True)
        untrained_results = lstm_main(train_data_file, w2v_model, testing=False, expt_name=EXPT_NAME, use_w2v=True)

    # get tf-idf vectorizer
    df = train_data_file
    vector_small = tf_idf_vectorizer_small(df[COMMENT_TEXT_INDEX])

    # get log regression score from tf-idf(2-6 n gram) log reg
    vector_big = tf_idf_vectorizer_big(df[COMMENT_TEXT_INDEX])
    aggressively_positive_model_report = build_logistic_regression_model(vector_big, df)


if __name__ == "__main__":
    import pickle

    summarized_sentence_data = pickle.load(SUM_SENTENCES_FILE, "rb")
    EXPT_NAME = "TEST"
    SAMPLE_DATA_FILE = './data/sample.csv'
    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    PREDICT_DATA_FILE = './data/predict.csv'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    print(EXPT_NAME)
    main(train_data_file=SAMPLE_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
         summarized_sentences=summarized_sentence_data, w2v_model=model, testing=True,
         expt_name=EXPT_NAME)

    """
    print("done with tests, loading true model")
    EXPT_NAME = "REAL"
    TRAIN_DATA_FILE = './data/train.csv'
    PREDICT_DATA_FILE = './data/predict.csv'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    main(train_data_file=TRAIN_DATA_FILE, predict_data_file=PREDICT_DATA_FILE,
         summarized_sentences=summarized_sentence_data, w2v_model=model, testing=False,
         expt_name=EXPT_NAME)
    """
