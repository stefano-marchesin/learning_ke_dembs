from elasticsearch import Elasticsearch


ES_HOST = {'host': 'localhost', 'port': 9200}
INDEX = 'ohsu_elastic'
DOC = 'abstract'
WORDS_FIELD = 'body_words'
ANALYZER = 'ohsu_analyzer'
RANKER = 'BM25'
STOPWORDS = 'path/to/stopwords/indri_stopwords.txt'
ES = Elasticsearch([ES_HOST])
