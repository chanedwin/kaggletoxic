import lda.datasets

from sklearn.feature_extraction.text import CountVectorizer


def get_lda_topics(sentences):
    vectorizer = CountVectorizer(stop_words='english')
    sentences = sentences.tolist()

    vectorizer.fit(sentences)
    tf_idf_sparse_matrix = vectorizer.transform(sentences)

    model = lda.LDA(n_topics=500, n_iter=10, random_state=1)
    topics = model.fit_transform(tf_idf_sparse_matrix)

    return model, topics


def predict_lda_topics(model, sentences):
    vectorizer = CountVectorizer(stop_words='english')

    vectorizer.fit(sentences)
    tf_idf_sparse_matrix = vectorizer.transform(sentences)

    topics = model.transform(tf_idf_sparse_matrix)
    return topics
