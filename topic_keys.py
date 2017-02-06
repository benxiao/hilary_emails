from gensim.models.wrappers.ldamallet import LdaMallet
import json


DATA_PATH = '../data/hilary'
result = []
for y in range(2009, 2012):
    for m in range(4):
        topic_keys = []
        model_path = DATA_PATH+'/mallet_files/enron_{}_{}_mallet_model'.format(y, m)
        lda = LdaMallet.load(model_path)
        for i in range(30): # num of topics
            topic_keys.append({w:str(p) for w, p in lda.show_topic(i, num_words=100)})
        result.append(topic_keys)

json.dump(result, open('topic_keys.json', 'w'))


