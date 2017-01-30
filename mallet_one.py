import json
from gensim import corpora
from gensim.models.wrappers.ldamallet import LdaMallet
from multiprocessing import cpu_count

MALLET_PATH='/usr/local/mallet-2.0.6/bin/mallet'
#DATA_PATH = '/Users/ranxiao/Desktop/data/arXiv'
# For debuggering


if __name__ == '__main__':
    lst = json.load(open('processed_2000_1.json'))
    dictionary = corpora.Dictionary(lst)
    dictionary.filter_extremes(5, 0.1)
    corpus = [dictionary.doc2bow(x) for x in lst]

    lda = LdaMallet(
        mallet_path=MALLET_PATH,
        corpus=corpus,
        id2word=dictionary,
        num_topics=30,
        optimize_interval=10,
        iterations=4000,
        workers=cpu_count(),
    )
