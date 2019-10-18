import glob
import os
import math
import string
import json
import itertools
import numpy as np
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from elasticsearch import helpers
from elasticsearch_dsl import Search, Q

from . import constants
from . import settings
from . import utils


class Index(object):
    """define an index instance and its associated methods"""

    def __init__(self):
        """initialize index variables"""
        self.es = constants.ES
        self.index = constants.INDEX
        self.analyzer = constants.ANALYZER
        self.doc = constants.DOC
        self.wfield = constants.WORDS_FIELD
        self.settings = settings.properties

    def index_corpus(self, corpus_path):  # @smarchesin TODO: generalize to different collections
        """index given corpus (word-based indexing)"""
        if self.es.indices.exists(index=self.index):
            print('index already exists')
        else:
            print('create index')
            self.es.indices.create(index=self.index, body=self.settings)
            # index corpus docs
            print('indexing... ')
            # create generator to iterate over docs
            i = ({'_index': self.index, '_type': self.doc, '_id': docno, '_source': {self.wfield: body}} for docno, body in utils.gen_doc(corpus_path))  # @smarchesin TODO: add self.index to gen_doc
            # index bulks of docs
            helpers.bulk(self.es, i)
            print('indexing finished!')
        return True

    def get_ix_terms(self):
        """return list of indexed terms"""
        terms = self.es.search(index=self.index, body={'aggs': {self.doc: {'terms': {'field': self.wfield, 'size': 999999}}}})
        return [term['key'] for term in terms['aggregations'][self.doc]['buckets']]

    def get_terms_stats(self):
        """get stats for indexed terms"""
        terms = self.get_ix_terms()
        synt_doc = ' '.join(terms)
        # get terms stats from es index
        terms_stats = self.es.termvectors(index=self.index, doc_type=self.doc, 
            term_statistics=True, field_statistics=False, positions=False, offsets=False, 
            body={'doc': {self.wfield: synt_doc}})
        # return terms stats
        return [(term, stats['doc_freq'], stats['ttf']) for term, stats in terms_stats['term_vectors'][self.wfield]['terms'].items()] 

    def get_doc_ids(self):
        """return list of doc ids"""
        s = Search(using=self.es, index=self.index, doc_type=self.doc)
        src = s.source([])
        return [h.meta.id for h in src.scan()]

    def get_doc_terms(self, doc_id):
        """return list of (positionally-ordered) doc terms given a doc id"""
        doc_terms = self.es.termvectors(index=self.index, doc_type=self.doc, fields=[self.wfield], id=doc_id, 
            positions=True, term_statistics=False, field_statistics=False, offsets=False)['term_vectors'][self.wfield]['terms']
        # get term positions within doc: {term: [pos1, pos2, ...]}
        doc_pos = {term: stats['tokens'] for term, stats in doc_terms.items()}
        # reverse doc_pos associating each position with the corresponding term
        terms_pos = {}
        for term, positions in doc_pos.items():
            for pos in positions:
                terms_pos[pos['position']] = term
        # return positionally-ordered doc terms
        return [terms_pos.get(i) for i in range(min(terms_pos), max(terms_pos) + 1) if terms_pos.get(i) != None]

    def analyze_query(self, query):
        """analyze query using index analyzer"""
        res = self.es.indices.analyze(index=self.index, body={'analyzer': self.analyzer, 'text': query})
        return [term['token'] for term in res['tokens']]

    def change_model(self, model, **kwargs):
        """change similarity model for current index"""
        model_settings = {'type': model}
        if kwargs is not None:
            for key, value in kwargs.items():
                model_settings[key] = value
        # close index before updating
        self.es.indices.close(index=self.index)
        # update settings
        similarity_settings = {'similarity': {'custom_model': model_settings}}
        self.es.indices.put_settings(index=self.index, body=similarity_settings)
        # re-open index after updating
        self.es.indices.open(index=self.index)
        return True

    def lexical_search(self, queries, qfield, rank_path, ranker):
        """perform search over queries using lexical models and return ranking"""
        out = open(rank_path + '/' + ranker + '.txt', 'w')
        print('searching over batch of {} queries'.format(len(queries)))
        # search queries
        for qid, qbody in tqdm(queries.items()):
            qres = self.es.search(index=self.index, size=1000, body={'query': {'match': {self.wfield: qbody[qfield]}}})
            for idx, rank in enumerate(qres['hits']['hits']):
                out.write('%s %s %s %d %f %s\n' % (qid, 'Q0', rank['_id'], idx, rank['_score'], ranker))
        out.close()
        return True
