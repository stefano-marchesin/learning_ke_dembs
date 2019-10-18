import os
import argparse
import json
import gensim
import multiprocessing
import numpy as np 

from tqdm import tqdm 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from gensim_utils import Utils, EpochRanker

flags = argparse.ArgumentParser()

flags.add_argument("--embs_size", default=200, type=int, help="The embedding dimension size.")
flags.add_argument("--epochs", default=15, type=int,
				   help="Number of epochs to train. Each epoch processes the training data once completely.")
flags.add_argument("--train_on_concepts", default=True, type=bool, 
				   help="Whether to train Doc2Vec or ConceptualDoc2Vec.")
flags.add_argument("--negative_samples", default=5, type=int, help="Negative samples per training example.")
flags.add_argument("--learning_rate", default=0.02, type=float, help="Learning rate.")
flags.add_argument("--linear_decay", default=0.0001, type=float, help="Minimum learning rate after linear decay.")
flags.add_argument("--window", default=8, type=int, help="The number of words to predict to the left/right of the target word.")
flags.add_argument("--min_count", default=5, type=int, help="Ignores all words with total frequency lower than this.")
flags.add_argument("--model_type", default=1, type=int, help="PV-DBOW: 0; PV-DM: 1.")
flags.add_argument("--minsize", default=1, type=int, help="Minimum word's size allowed.")
flags.add_argument("--remove_digits", default=False, help="Whether to remove digits-only tokens from corpus.")
flags.add_argument("--stopwords", default=False, type=bool,
				   help="Whether to remove stopwords from corpus (Indri stopword list applied).")
flags.add_argument("--stypes_fname", default="/home/ims/Desktop/Marchesin/SAFIR/semantic_types.txt", type=str, help="Path to semantic types file.")
flags.add_argument("--threshold", default=0.7, type=float, help="Minimum similarity value between strings for QuickUMLS." )
flags.add_argument("--seed", default=0, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--corpus", default="OHSUMED_ALLTYPES", type=str, help="Corpus to use.")
flags.add_argument("--query_fname", default="topics_orig", type=str, help="Queries file name.")
flags.add_argument("--qrels_fname", default="qrels_short", type=str, help="Qrels file name.")
flags.add_argument("--query_field", default="title", type=str, help="Query field to consider when performing retrieval.")
flags.add_argument("--reference_measure", default="P_10", type=str, help="Reference measure to optimize.")
flags.add_argument("--model_name", default="cdoc2vec_embs_200_ohsu", type=str, help="Model name.")

FLAGS = flags.parse_args()


class Options(object):
	"""options used by the Neural Vector Space Model (NVSM)"""

	def __init__(self):
		# embeddings dimension
		self.embs_size = FLAGS.embs_size
		# epochs to train
		self.epochs = FLAGS.epochs
		# train on concepts
		self.train_on_concepts = FLAGS.train_on_concepts
		# number of negative samples per example
		self.neg_samples = FLAGS.negative_samples
		# learning rate
		self.learn_rate = FLAGS.learning_rate
		# linear decay
		self.lin_decay = FLAGS.linear_decay
		# window size
		self.window = FLAGS.window
		# min count
		self.min_count = FLAGS.min_count
		# model
		self.model_type = FLAGS.model_type
		# minimum word size
		self.minsize = FLAGS.minsize
		# remove digits
		self.remove_digits = FLAGS.remove_digits
		# stopwords considered
		self.stopwords = FLAGS.stopwords
		# semantic types filename
		self.stypes_fname = FLAGS.stypes_fname
		# threshold
		self.threshold = FLAGS.threshold
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
		# reference measure
		self.ref_measure = FLAGS.reference_measure
		# model name
		self.model_name = FLAGS.model_name


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set folders
	corpus_folder = 'corpus/' + opts.corpus_name + '/' + opts.corpus_name
	index_folder = 'corpus/' + opts.corpus_name + '/index'
	model_folder = 'corpus/' + opts.corpus_name + '/models/' + opts.model_name
	data_folder = 'corpus/' + opts.corpus_name + '/data'
	query_folder = 'corpus/' + opts.corpus_name + '/queries'
	qrels_folder = 'corpus/' + opts.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + opts.corpus_name + '/rankings/' + opts.model_name

	# create folders
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)
	if not os.path.exists(index_folder):
		os.makedirs(index_folder)
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# initialize utils funcs
	utils = Utils(opts.seed)

	# compute available number of CPUs
	cpu_count = multiprocessing.cpu_count()

	# load stop words
	if opts.stopwords:
		# set stopwords
		stops_fname = './indri_stopwords.txt'
		stops = utils.load_stopwords(stopwords_fname)
	else:
		stops = None

	# load queries
	print('load {} queries'.format(opts.corpus_name))
	queries = utils.read_queries(query_folder + '/' + opts.qfname)

	"""MODEL PRE-PROCESSING"""
	if opts.train_on_concepts:  # pre process data to train ConceptualDoc2Vec
		print('--train_on_concepts = {}. Training ConceptualDoc2Vec'.format(opts.train_on_concepts))
		if not os.path.exists(data_folder + '/cdocs.json'):

			if not os.path.exists(data_folder + '/docs.json'):
				# tokenize corpus into list of lists and return dict {docno: doc_tokens}
				corpus = utils.tokenize_corpus(corpus_folder, stopwords=stops, out_path=data_folder, remove_digits=opts.remove_digits, minsize=opts.minsize)
				# buil word dictionary
				token_dict = utils.build_vocab(corpus.values(), min_cut_freq=opts.min_count, out_path=data_folder)
			else:
				# load required data
				with open(data_folder + '/docs.json', 'r') as cf:
					corpus = json.load(cf)
				with open(data_folder + '/word_dict.json', 'r') as wdf:
					token_dict = json.load(wdf)
			
			# associate to each term within term_dict the corresponding CUI within UMLS (obtained through QuickUMLS)
			if not os.path.exists(data_folder + '/term2cui.json'):
				print('map CUIs to terms within vocabulary')
				term2cui = utils.get_term2cui(token_dict, data_folder, threshold=opts.threshold, stypes_fname=opts.stypes_fname)
			else:
				# load term2cui data
				print('load {term: cui} dictionary')
				with open(data_folder + '/term2cui.json', 'r') as tcf:
					term2cui = json.load(tcf)
			# keep only CUIs that present an entry in the MeSH lexicon 
			print('keep only CUIs presenting an entry in MeSH lexicon')
			term2cui = utils.cui2source(term2cui, source='MSH')
			# convert corpus from words to concepts and build concept dictionary
			print('convert docs from words to (MeSH) CUIs')
			corpus = utils.conceptualize_corpus(corpus, token_dict, term2cui, data_folder)
			# build concept dictionary - min_cut_freq = 1 as words have been already cut before being converted into CUIs
			token_dict = utils.build_cvocab(corpus.values(), 1, data_folder)
		
		else: 
			# load required data
			with open(data_folder + '/cdocs.json', 'r') as cf:
				corpus = json.load(cf)
			with open(data_folder + '/concept_dict.json', 'r') as cdf:
				token_dict = json.load(cdf)

			# load term2cui data
			print('load {term: cui} dictionary')
			with open(data_folder + '/term2cui.json', 'r') as tcf:
				term2cui = json.load(tcf)
			# keep only CUIs that present an entry in the MeSH lexicon 
			print('keep only CUIs presenting an entry in MeSH lexicon')
			term2cui = utils.cui2source(term2cui, source='MSH')

		# load word dictionary to process queries and convert words to concepts
		with open(data_folder + '/word_dict.json', 'r') as wdf:
			word_dict = json.load(wdf)
		# convert the target query field from words to (MESH) CUIs
		print('convert queries from words to (MeSH) CUIs')
		queries = utils.conceptualize_queries(queries, opts.qfield, word_dict, term2cui)
		# delete word_dict to free memory space
		del word_dict

	else:  # pre process data to train Doc2Vec
		print('--train_on_concepts = {}. Training Doc2Vec'.format(opts.train_on_concepts))
		if not os.path.exists(data_folder + '/docs.json'):
			# tokenize corpus into list of lists and return dict {docno: doc_tokens}
			corpus = utils.tokenize_corpus(corpus_folder, stopwords=stops, out_path=data_folder, remove_digits=opts.remove_digits, minsize=opts.minsize)
			# buil word dictionary
			token_dict = utils.build_vocab(corpus.values(), min_cut_freq=opts.min_count, out_path=data_folder)
		
		else:
			# load required data
			with open(data_folder + '/docs.json', 'r') as cf:
				corpus = json.load(cf)
			with open(data_folder + '/word_dict.json', 'r') as wdf:
				token_dict = json.load(wdf)

	# prepare corpus for Doc2Vec models - require docs in the form of TaggedDocument
	print('prepare corpus for Doc2Vec models - it requires docs in TaggedDocument format')
	corpus = [TaggedDocument(words=[token for token in doc if token in token_dict], tags=[docno]) for docno, doc in tqdm(corpus.items()) if doc]

	# initialize Doc2Vec/ConceputalDoc2Vec model
	print('initialize model')
	model = Doc2Vec(epochs=opts.epochs, vector_size=opts.embs_size, window=opts.window, alpha=opts.learn_rate, min_alpha=opts.lin_decay, min_count=1, 
					dm=opts.model_type, dm_mean=1, dbow_words=1, negative=opts.neg_samples, seed=opts.seed, sample=None, workers=cpu_count)
	
	# build Doc2Vec/ConceptualDoc2Vec vocabulary
	print('build model vocabulary')
	model.build_vocab(corpus)

	# sanity check on model vocab - model.wv.vocab must be equal to token_dict
	token_dict.pop('UNK', None)
	assert set(model.wv.vocab.keys()) == set(token_dict.keys())

	# dispaly model vocabulary stats
	print('number of unique tokens: {}'.format(len(model.wv.vocab)))

	# initialize callback class to perform ranking after each training epoch
	epoch_ranker = EpochRanker(corpus, queries, None, utils, True, opts, model_folder, rankings_folder, qrels_folder)

	# train and evaluate model in terms of IR effectiveness
	print('train the model for {} epochs and evaluate it in terms of IR effectiveness'.format(opts.epochs))
	model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[epoch_ranker])
	print('model training finished!')

	# get best model in terms of reference measure
	best_epoch = epoch_ranker.best_epoch
	best_score = epoch_ranker.best_score
	print('best model found at epoch {} with {}: {}'.format(best_epoch, opts.ref_measure, best_score))


if __name__ == "__main__":
	main() 