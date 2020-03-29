import os
import argparse
import operator
import gensim
import umls
import pytrec_eval
import numpy as np

from tqdm import tqdm
from elasticsearch import Elasticsearch 
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec

from lexical_baselines.es_utils.index import Index
from gensim_utils import Utils


flags = argparse.ArgumentParser()
flags.add_argument("--num_top_docs", default=3, type=int, help="Number of top k docs to consider for query expansion.")
flags.add_argument("--num_top_words_per_doc", default=100, type=int, help="Number of top n words per doc.")
flags.add_argument("--num_top_words", default=5, type=int, help="Number of top m words to consider for query expansion.")
flags.add_argument("--beta", default=0.6, type=float, help="Regularization parameter to retrofit doc vectors using Doc2Vec and ConceptualDoc2Vec models.")
flags.add_argument("--seed", default=0, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--corpus", default="OHSUMED_ALLTYPES", type=str, help="Corpus to use.")
flags.add_argument("--query_fname", default="topics_orig", type=str, help="Queries file name.")
flags.add_argument("--qrels_fname", default="qrels_short", type=str, help="Qrels file name.")
flags.add_argument("--query_field", default="title", type=str, help="Query field to consider when performing retrieval.")
flags.add_argument("--txt_d2v_model", default="/home/ims/Desktop/Marchesin/SAFIR/corpus/OHSUMED_ALLTYPES/models/doc2vec_embs_200_ohsu/doc2vec_embs_200_ohsu12.model", type=str, help="Text-based Doc2Vec model used to perform query expansion.")
flags.add_argument("--concept_d2v_model", default="/home/ims/Desktop/Marchesin/SAFIR/corpus/OHSUMED_ALLTYPES/models/cdoc2vec_embs_200_ohsu/cdoc2vec_embs_200_ohsu2.model", type=str, help="Concept-based Doc2Vec model used to perform query expansion.")
flags.add_argument("--retro_model", default="/home/ims/Desktop/Marchesin/SAFIR/corpus/OHSUMED_ALLTYPES/models/retro_embs_200_ohsu/retro_alpha0.01_beta0.6_init-uniform_normFalse_model.npy", type=str, help="Retrofitted Doc2Vec model used to perform query expansion.")
flags.add_argument("--bow_model", default="/home/ims/Desktop/Marchesin/SAFIR/corpus/OHSUMED/rankings/lexical_baselines/BM25_title_orig.txt", type=str, help="Bow-of-Words model run.")
flags.add_argument("--model_name", default="exp_d3w5_doc2vec_ohsu", type=str, help="Expansion run name.")
FLAGS = flags.parse_args()


class Options(object):
	"""options used by the Doc2Vec model"""

	def __init__(self):
		# number of top docs
		self.num_top_docs = FLAGS.num_top_docs
		# number of top words per doc
		self.num_top_words_per_doc = FLAGS.num_top_words_per_doc
		# number of top words per query
		self.num_top_words = FLAGS.num_top_words
		# beta
		self.beta = FLAGS.beta
		# seed
		self.seed = FLAGS.seed
		# corpus name
		self.corpus_name = FLAGS.corpus
		# query file name
		self.qfname = FLAGS.query_fname
		# qrels file name
		self.qrels_fname = FLAGS.qrels_fname 
		# query field
		self.qfield = FLAGS.query_field
		# text-based doc2vec model
		self.txt_d2v_model_path = FLAGS.txt_d2v_model
		# concept-based doc2vec model
		self.concept_d2v_model_path = FLAGS.concept_d2v_model
		# retrofitted doc2vec model
		self.retro_model_path = FLAGS.retro_model
		# BoW model
		self.bow_model_path = FLAGS.bow_model
		# qexp model name
		self.model_name = FLAGS.model_name


def top_words_of_doc(d2v_model, doc_id, topN=100):
	"""return topN words from doc given its doc_id and the d2v_model"""
	return d2v_model.most_similar([d2v_model.docvecs[doc_id]], topn=topN)


def top_words_of_vector(d2v_model, vector, topN=100):
	"""return topN words given a (inferred) vector"""
	return d2v_model.most_similar([vector], topn=topN)


def read_ranking(bow_model_path):
	"""return BoW ranking as dict of dicts {qid: {doc_id: score, ...}, ...}"""
	print('read BoW ranking')
	with open(bow_model_path, 'r') as f:
		# pytrec_eval loads ranking as dict of dicts
		run = pytrec_eval.parse_run(f)
	return run


def get_query_top_docs(bow_run, qid, top_n_docs):
	"""return topN doc ids given a run and a query id"""
	doc_ids_and_scores = bow_run[qid]
	doc_ids = sorted(doc_ids_and_scores.keys(), key=operator.itemgetter(1), reverse=True)
	return doc_ids[:top_n_docs]


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set folders
	query_folder = 'corpus/' + opts.corpus_name + '/queries'
	qrels_folder = 'corpus/' + opts.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + opts.corpus_name + '/rankings/' + opts.model_name

	# create folders
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# load utils functions - set random seed
	utils = Utils(opts.seed)
	# load UMLS lookup functions
	umls_lookup = umls.UMLSLookup()

	# load queries
	print('load {} queries'.format(opts.corpus_name))
	queries = utils.read_queries(query_folder + '/' + opts.qfname)

	# load BoW run
	bow_model = read_ranking(opts.bow_model_path)

	# load models
	print('load models')
	txt_d2v_model = gensim.models.Doc2Vec.load(opts.txt_d2v_model_path)
	concept_d2v_model = gensim.models.Doc2Vec.load(opts.concept_d2v_model_path)
	retro_model = np.load(opts.retro_model_path, allow_pickle=True).item()

	##### QUERY EXPANSION #####

	N_top_docs = opts.num_top_docs
	N_top_words_doc = opts.num_top_words_per_doc
	N_top_words_query= opts.num_top_words

	queries_t = {}
	queries_c = {}
	queries_r = {}

	print('perform query expansion for each model')
	for qid, qtext in tqdm(queries.items()):

		# get N_top_docs for given query
		print('get top {} docs for query {}'.format(N_top_docs, qid))
		query_top_docs = get_query_top_docs(bow_model, qid, N_top_docs)

		'''
		for each doc in query_top_docs, pick N_top_words_doc, add to pool
		sort the pool, get N_top_words_query to add in given query
		'''

		top_concept_t = {}
		top_concept_c = {}
		top_concept_r = {}

		for top_doc_id in query_top_docs:
			top_t = top_words_of_doc(txt_d2v_model, top_doc_id, N_top_words_doc)

			if top_doc_id in concept_d2v_model.docvecs:
				top_c = top_words_of_doc(concept_d2v_model, top_doc_id, N_top_words_doc)

			if top_doc_id in retro_model: 
				top_retro_doc = retro_model[top_doc_id]
				if opts.beta < 0.5:  # prioritize concepts 
					top_r = top_words_of_vector(concept_d2v_model, top_retro_doc, N_top_words_doc)
				else:  # prioritize words
					top_r = top_words_of_vector(txt_d2v_model, top_retro_doc, N_top_words_doc)
			
			for i in range(N_top_words_doc):
				if len(top_t) == N_top_words_doc:  # doc_id found by txt_d2v_model
					term = top_t[i][0]
					score = top_t[i][1]
					if term in top_concept_t:  # combsum
						top_concept_t[term] += score
					else:
						top_concept_t[term] = score

				if len(top_c) == N_top_words_doc:  # doc_id found by concept_d2v_model
					term = top_c[i][0]
					score = top_c[i][1]
					if term in top_concept_c:  # combsum
						top_concept_c[term] += score
					else:
						top_concept_c[term] = score

				if len(top_r) == N_top_words_doc:  # doc_id found by retro_model
					term  = top_r[i][0]
					score = top_r[i][1]
					if term in top_concept_r:  # combsum
						top_concept_r[term] += score
					else:
						top_concept_r[term] = score
				
		# sorting top_concept lists
		sorted_candidates_t = sorted(top_concept_t.items(), key=operator.itemgetter(1))  # [(id1,min_sim), ... , (idn, max_sim)]
		sorted_candidates_c = sorted(top_concept_c.items(), key=operator.itemgetter(1))  # [(id1,min_sim), ... , (idn, max_sim)]
		sorted_candidates_r = sorted(top_concept_r.items(), key=operator.itemgetter(1))  # [(id1,min_sim), ... , (idn, max_sim)]
		top_term_t = sorted_candidates_t[-N_top_words_query:]
		top_term_c = sorted_candidates_c[-N_top_words_query:]
		top_term_r = sorted_candidates_r[-N_top_words_query:]

		query_new_t = qtext[opts.qfield]
		query_new_c = qtext[opts.qfield]
		query_new_r = qtext[opts.qfield]

			# query_new_t = ''
			# query_new_c = ''
			# query_new_r = ''

		count_t = 0
		count_c = 0
		count_r = 0

		for term, _ in top_term_t:
			query_new_t += ' ' + term
			count_t += 1
		
		for cui, _ in top_term_c:
			cui = cui.upper()
			term_variants = [term_and_source for term_and_source in umls_lookup.lookup_synonyms(cui=cui, preferred=True) if term_and_source[1] == 'MSH']
			term = term_variants[0][0]  # preferred term
			query_new_c += ' ' + term
			count_c += 1

		if opts.beta < 0.5:
			for cui, _ in top_term_r:
				cui = cui.upper()
				term_variants = [term_and_source for term_and_source in umls_lookup.lookup_synonyms(cui=cui, preferred=True) if term_and_source[1] == 'MSH']
				term = term_variants[0][0]
				query_new_r += ' ' + term
				count_r += 1
		else: 
			for term, _ in top_term_r:
				query_new_r += ' ' + term
				count_r += 1 
		
		queries_t[qid] = {opts.qfield: query_new_t}
		queries_c[qid] = {opts.qfield: query_new_c}
		queries_r[qid] = {opts.qfield: query_new_r}

	es = Elasticsearch([{'host': 'localhost', 'port':9200}])
	# set Index instance
	ix = Index()

	print('search and evaluate text-based doc2vec query expansion')
	# perform lexical search over given query field w/ chosen model
	ix.lexical_search(queries_t, opts.qfield, rankings_folder, opts.model_name + '_txt_d2v') 
	# evaluate performed search
	scores = utils.evaluate(['recall.20', 'P_20', 'map'], rankings_folder, opts.model_name + '_txt_d2v', qrels_folder, opts.qrels_fname)

	print('search and evaluate concept-based doc2vec query expansion')
	# perform lexical search over given query field w/ chosen model
	ix.lexical_search(queries_c, opts.qfield, rankings_folder, opts.model_name + '_concept_d2v') 
	# evaluate performed search
	scores = utils.evaluate(['recall.20', 'P_20', 'map'], rankings_folder, opts.model_name + '_concept_d2v', qrels_folder, opts.qrels_fname)

	print('search and evaluate retrofitted doc2vec query expansion')
	# perform lexical search over given query field w/ chosen model
	ix.lexical_search(queries_r, opts.qfield, rankings_folder, opts.model_name + '_retro_d2v') 
	# evaluate performed search
	scores = utils.evaluate(['recall.20', 'P_20', 'map'], rankings_folder, opts.model_name + '_retro_d2v', qrels_folder, opts.qrels_fname)


if __name__ == "__main__":
	main()
