import lda.datasets
import numpy as np

from utils import load_data, DATA_FILE, COMMENT_TEXT_INDEX
from sklearn.feature_extraction.text import CountVectorizer

df = load_data(DATA_FILE)

sentences = df[COMMENT_TEXT_INDEX]

vectorizer = CountVectorizer(stop_words='english')
sentences = sentences.tolist()

vectorizer.fit(sentences)
tf_idf_sparse_matrix = vectorizer.transform(sentences)
print(vectorizer.get_feature_names())

print(tf_idf_sparse_matrix)

model = lda.LDA(n_topics=500, n_iter=10, random_state=1)
model.fit(tf_idf_sparse_matrix)

topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    topic_words = vectorizer.get_feature_names()[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
