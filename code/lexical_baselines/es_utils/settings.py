from . import utils
from . import constants


# define settings and mapping used to index with es
properties = {
    'settings': {
        'number_of_shards': '1',
        'number_of_replicas': '0',
        "index": {
            "blocks": {
                "read_only_allow_delete": "false"
            }
        },
        'analysis': {
            'filter': {
                'custom_stopwords': {
                    'type': 'stop',
                    'stopwords': utils.load_stopwords('./' + constants.STOPWORDS)
                },
                'length_filter': {
                    'type': 'length',
                    'min': 3
                },
                'possessive_stemmer': {
                    'type': 'stemmer',
                    'language': 'possessive_english'
                },
                'porter_stemmer': {
                    'type': 'stemmer',
                    'language': 'english'
                }
            },
            'analyzer': {
                constants.ANALYZER: {
                    'tokenizer': 'classic',
                    'filter': [
                        #'possessive_stemmer',
                        'lowercase',
                        #'length_filter',
                        #'custom_stopwords',
                        #'porter_stemmer'
                    ]
                }
            }
        },
        'similarity': {
            'custom_model': {
                'type': constants.RANKER
            }
        }
    },
    'mappings': {
        constants.DOC: {
            'properties': {
                constants.WORDS_FIELD: {
                    'type': 'text',
                    'similarity': 'custom_model',
                    'analyzer': constants.ANALYZER
                }, 
                constants.CONCEPTS_FIELD:{
                    'type': 'keyword',
                    'similarity': 'custom_model',
                    'norms': 'true',
                    'split_queries_on_whitespace': 'true'
                }
            }
        }
    }
}
