import gensim

from constants.general import DATA_DIR, NO_OF_DATA
import pandas
from gensim.corpora import Dictionary
from gensim import corpora, models
from pprint import pprint
import nltk

from functions.functions import preprocess, lemmatize_stemming

# nltk.download('wordnet')

if __name__ == '__main__':

    result = pandas.DataFrame(columns=['index', 'topic'])
    df = pandas.read_csv(DATA_DIR + "parkman_sentiment.csv", sep=";")
    headlines = df[['content']]
    headlines['index'] = headlines.index
    documents = headlines  # a list indexed of documents(sentences)

    # doc_sample = documents[documents['index'] == 4310].values[0][0]

    # print(documents[(documents['index'] < 10)]['headline_text'])

    processed_docs = documents[(documents['index'] < len(df))]['content'].map(preprocess)  # pos tagging and lemmentize

    dictionary = Dictionary(processed_docs)  # Dictionary of unique words from the document
    dictionary.filter_extremes(no_below=15,
                               no_above=0.5)  # filtering out word counts in the range 0.5 < n < 15 and keeping the top 1000 words left

    bow_corpus = [dictionary.doc2bow(doc) for doc in
                  processed_docs]  # creating a corpus containing a bag of words(BOW) for each document

    tfidf = models.TfidfModel(bow_corpus)  # giving a statistical score/weight to each word in the corpus
    corpus_tfidf = tfidf[bow_corpus]

    lda_model = models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2,
                                    workers=2)  # LDA model using the BOW corpus as training data
    lda_model_tfidf = models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2,
                                          workers=4)  # LDA model using the TFIDF corpus as training data

    # print(lda_model_tfidf.print_topics(-1))

    for idx, topic in lda_model_tfidf.print_topics(-1):
        result = result.append({'index': idx + 1, 'topic': topic}, ignore_index=True)

    result.to_csv("./resources/results/parkman_topics.csv", index=False)

    print(result)
