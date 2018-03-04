from lstm_model import lstm_main
from utils import load_w2v_model_from_path


def main(data_file, w2v_model, testing, expt_name="test"):
    if testing:
        pre_trained_results = lstm_main(data_file, w2v_model, testing, expt_name=expt_name, use_w2v=True)
        untrained_results = lstm_main(data_file, w2v_model, testing, expt_name=expt_name, use_w2v=False)
    else:
        pre_trained_results = lstm_main(data_file, w2v_model, testing=False, expt_name=EXPT_NAME, use_w2v=True)
        untrained_results = lstm_main(data_file, w2v_model, testing=False, expt_name=EXPT_NAME, use_w2v=True)


if __name__ == "__main__":
    EXPT_NAME = "TEST"
    SAMPLE_DATA_FILE = './data/sample.csv'
    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    print(EXPT_NAME)
    """
    main(SAMPLE_DATA_FILE, model, testing=True, expt_name=EXPT_NAME)

    """
    print("done with tests, loading true model")
    EXPT_NAME = "REAL"
    DATA_FILE = './data/train.csv'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    main(DATA_FILE, model, testing=False, expt_name=EXPT_NAME)
