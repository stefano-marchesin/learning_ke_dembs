import os
import gensim
import sys
import argparse
import numpy as np 

from tqdm import tqdm
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine


flags = argparse.ArgumentParser()

flags.add_argument("--doc2vec", default="/home/ims/Desktop/Marchesin/SAFIR/corpus/OHSUMED_ALLTYPES/models/doc2vec_embs_200_ohsu/doc2vec_embs_200_ohsu12.model", type=str, 
				   help="Path to Doc2Vec model.")
flags.add_argument("--cdoc2vec", default="/home/ims/Desktop/Marchesin/SAFIR/corpus/OHSUMED_ALLTYPES/models/cdoc2vec_embs_200_ohsu/cdoc2vec_embs_200_ohsu2.model", type=str, 
				   help="Path to ConceptualDoc2Vec model.")
flags.add_argument("--epochs", default=500, type=int, help="Maximum number of iterations considered to perform retrofitting.")
flags.add_argument("--init", default="uniform", type=str, help="Initialization strategy for retrofitted doc vectors. Choose between 'unif' and 'txt_d2v'.")
flags.add_argument("--norm", default=False, type=bool, help="Whether to normalize doc vectors prior retrofitting.")
flags.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate used to optimize retrofitted doc representations.")
flags.add_argument("--beta", default=0.6, type=float, help="Regularization parameter to retrofit doc vectors using Doc2Vec and ConceptualDoc2Vec models.")
flags.add_argument("--min_loss", default=1e-7, type=float, help="Minimum loss value after which the optimization process stops for the target document representation.")
flags.add_argument("--seed", default=42, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--qrels_fname", default="qrels_short", type=str, help="Qrels filename.")
flags.add_argument("--query_fname", default="topics_orig", type=str, help="Query filename.")
flags.add_argument("--query_field", default="title", type=str, help="Query field to consider for retrieval.")
flags.add_argument("--reference_measure", default="P_10", type=str, help="Reference measure to consider for optimization.")
flags.add_argument("--corpus_name", default='OHSUMED_ALLTYPES', type=str, help="Corpus to consider.")
flags.add_argument("--model_name", default='retro_embs_200_ohsu', type=str, help="Model name.")

FLAGS = flags.parse_args()


class Options(object):
	"""options used by Doc2Vec and ConceptualDoc2Vec models"""
	def __init__(self):
		# doc2vec path
		self.doc2vec = FLAGS.doc2vec
		# cdoc2vec path
		self.cdoc2vec = FLAGS.cdoc2vec
		# number of epochs
		self.epochs = FLAGS.epochs
		# initialization strategy
		self.init = FLAGS.init
		# normalization 
		self.norm = FLAGS.norm
		# learning rate
		self.alpha = FLAGS.learning_rate
		# beta
		self.beta = FLAGS.beta
		# minimum loss
		self.min_loss = FLAGS.min_loss
		# seed
		self.seed = FLAGS.seed
		# corpus name
		self.corpus_name = FLAGS.corpus_name
		# qrels file name
		self.qrels_fname = FLAGS.qrels_fname 
		# query file name
		self.qfname = FLAGS.query_fname
		# query field
		self.qfield = FLAGS.query_field
		# reference measure
		self.ref_measure = FLAGS.reference_measure
		# model name
		self.model_name = FLAGS.model_name



def test_docvecs(txt_d2v_model_path, concept_d2v_model_path, retro_model_path):
	"""compute the cosine similarity between text_d2v, concept_d2v and retrofitted"""
	t_model = gensim.models.Doc2Vec.load(txt_d2v_model_path)
	c_model = gensim.models.Doc2Vec.load(concept_d2v_model_path)
	r_model = np.load(retro_model_path).item()
	# select first docno for testing purposes
	docno = r_model.keys()[0]
	# get doc embedding for given docno
	t = t_model.docvecs[docno]
	c = c_model.docvecs[docno]
	r = r_model[docno]
	# compute cosine similarity (1 - cosine_distance)
	print(1 - cosine(t, c))
	print(1 - cosine(t, r))
	return True


def z_normalize(vector):
	"""perform z-normalization over vector"""
	mean = np.mean(vector)
	std = np.std(vector)
	if std != 0:
		vector = (vector - mean) / std
	else: 
		vector = vector - vector
	return vector


def compute_loss(x, x_txt, x_concept, beta):
	"""compute loss and gradients for given representations"""
	if x_txt is not None and x_concept is not None:
		loss = beta*euclidean(x, x_txt)**2 + (1-beta)*euclidean(x, x_concept)**2
		grad = beta*2*(x - x_txt) + (1-beta)*2*(x - x_concept)
		return (loss, grad)
	elif x_txt is not None:
		loss = beta*euclidean(x, x_txt)**2
		grad = beta*2*(x - x_txt)
		return (loss, grad)
	elif x_concept is not None:
		loss = (1-beta)*euclidean(x, x_concept)**2
		grad = (1-beta)*2*(x - x_concept)
		return (loss, grad)
	else:
		return None, None


def retrofit(txt_d2v_model_path, concept_d2v_model_path, out_dir, init='unif', norm=False, alpha=0.01, beta=0.6, min_loss=1e-7, epochs=500):
	"""retrofit doc vectors using word- and concept-based doc vectors"""
	txt_d2v_model = gensim.models.Doc2Vec.load(txt_d2v_model_path)
	concept_d2v_model = gensim.models.Doc2Vec.load(concept_d2v_model_path)
	# sanity check on doc lengths
	try: 
		txt_size = len(txt_d2v_model.docvecs[0])
		concept_size = len(concept_d2v_model.docvecs[0])
		assert txt_size == concept_size
	except Exception as e:
			print('inconsistent vector size between the 2 models: {} vs {}'.format(txt_size, concept_size))
	# get docnos from txt_d2v_model
	docnos = txt_d2v_model.docvecs.doctags.keys()
	
	# vector inizialization for retrofitted docs
	retro_docs = {}
	for docno in docnos:
		retro_docs[docno] = np.random.uniform(-1, 1, txt_size)
		if init == 'txt_d2v':  # initialize doc vectors as txt_d2v_model doc vectors
			retro_docs = txt_d2v_model.docvecs[docno]
		if norm:  # normalize doc vectors prior retrofitting
			retro_docs[docno] = z_normalize(retro_docs[docno])

	# start retrofitting
	for epoch in tqdm(range(0, epochs)):
		epoch_loss = 0.0
		count = 0
		for docno in docnos:
			# get doc vectors for given docno
			x = retro_docs[docno]
			x_t = None
			if docno in txt_d2v_model.docvecs:
				x_t = txt_d2v_model.docvecs[docno]
			x_c = None
			if docno in concept_d2v_model.docvecs:
				x_c = concept_d2v_model.docvecs[docno]
			# compute loss and return gradients
			loss, grad = compute_loss(x, x_t, x_c, beta)
			# update loss and count
			epoch_loss += loss
			count += 1
			# update doc vectors w/ gradients
			x = x - (alpha * grad)
			retro_docs[docno] = x
		# compute epoch loss
		epoch_loss = epoch_loss / count
		print('loss {} at epoch {}'.format(epoch_loss, epoch + 1))
		if epoch_loss < min_loss:
			print('loss is smaller than {}. stop retrofitting.'.format(min_loss))
			break

	# store doc vectors
	print('store retrofitted doc vectors in {}'.format(out_dir))
	save_path = os.path.join(out_dir, 'retro_alpha{}_beta{}_init-{}_norm{}_model.npy'.format(alpha, beta, init, norm))
	np.save(save_path, retro_docs)
	return True


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set model folder
	model_folder = 'corpus/' + opts.corpus_name + '/models/' + opts.model_name
	# create model folder 
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)

	# retrofit doc vectors for opts.epochs or until loss == opts.min_loss
	print('retrofit doc vectors for {} iterations or until loss == {} '.format(opts.epochs, opts.min_loss))
	retrofit(opts.doc2vec, opts.cdoc2vec, model_folder, init=opts.init, alpha=opts.alpha, beta=opts.beta, min_loss=opts.min_loss, epochs=opts.epochs)
	

if __name__ == "__main__":
	main()