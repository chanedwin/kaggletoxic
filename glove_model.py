from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def convert_glove_model_to_w2v_model(glove_model_path, w2v_model_path):
    """
    :param glove_model_path:
    :type glove_model_path:
    :param w2v_model_path:
    :type w2v_model_path:
    """
    glove2word2vec(glove_input_file=glove_model_path, word2vec_output_file=w2v_model_path)


def load_w2v_model_from_path(model_path, binary_input=False):
    """
    :param model_path: path to w2v model
    :type model_path: string
    :param binary_input: True : binary input, False : text input
    :type binary_input: boolean
    :return: loaded w2v model
    :rtype: KeyedVectors object
    """
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=binary_input)
    return w2v_model

if __name__ == "__main__" :
    input = "/home/edwin/projects/kaggletoxic/models/glove.840B.300d.txt"
    output = "/home/edwin/projects/kaggletoxic/models/w2v.840B.300d.txt"
    convert_glove_model_to_w2v_model(input,output)
