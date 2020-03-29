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
flags.add_argument("--model_name", default="retro_doc2vec_ohsu", type=str, help="Expansion run name.")
FLAGS = flags.parse_args()


class Options(object):
	"""options used by the Doc2Vec and ConceptualDoc2Vec models"""

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


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# set options
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
	
	# load queries
	print('load {} queries'.format(opts.corpus_name))
	queries = utils.read_queries(query_folder + '/' + opts.qfname)

	# load models
	print('load models')
	if opts.beta < 0.5:
		d2v_model = gensim.models.Doc2Vec.load(opts.concept_d2v_model_path)
	else:
		d2v_model = gensim.models.Doc2Vec.load(opts.txt_d2v_model_path)
	retro_model = np.load(opts.retro_model_path, allow_pickle=True).item()

	# compute doc embs
	print('get document embeddings')
	doc_embs = np.array(list(retro_model.values()))
	doc_ids = np.array(list(retro_model.keys()))
	
	q_ids = list()
	q_embs = list()
	# loop over queries and generate query embs
	for qid, qtext in queries.items():
		# compute query emb as the sum of its word embs
		q_emb = utils.query2emb(qtext[opts.qfield], d2v_model)
		if q_emb is None:
			print('query {} does not contain known terms'.format(qid))
		else:
			q_embs.append(q_emb)
			q_ids.append(qid)
	# convert q_embs to numpy
	q_embs = np.array(q_embs)

	# perform semantic search with trained embs
	utils.semantic_search(doc_ids, doc_embs, q_ids, q_embs, rankings_folder, opts.model_name)
	# evaluate ranking list
	scores = utils.evaluate(['recall.20', 'P_20', 'map'], rankings_folder, opts.model_name, qrels_folder, opts.qrels_fname)


if __name__ == "__main__":
	main()
