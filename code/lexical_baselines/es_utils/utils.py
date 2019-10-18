import math
import subprocess
import glob
import numpy as np
import xml.etree.ElementTree as ETree

from tqdm import tqdm


def load_stopwords(stopwords_path):
        """read stopwords file into list"""
        with open(stopwords_path, 'r') as sl:
            stop_words = [stop.strip() for stop in sl]
        return stop_words


def load_semtypes(semtypes_path):
    """read semantic types into list"""
    with open(semtypes_path, 'r') as st:
        semtypes = [semtype.split('|')[1] for semtype in st]
    return semtypes


def get_trec_corpus(corpus_path):
    """convert trec style corpus into (valid) xml"""
    docs = glob.glob(corpus_path + '/**/*.txt', recursive=True)
    for doc in docs:
        with open(doc, 'r') as f:  # read doc
            xml = f.read()
        # convert into true xml
        xml = '<ROOT>' + xml + '</ROOT>'
        # fix bad-formed tokens
        xml = xml.replace('&', '&amp;')
        yield xml


def gen_doc(corpus_path):  # @smarchesin TODO: make it general for collections other than OHSUMED
    """generate doc from batch of docs"""
    ohsu = get_trec_corpus(corpus_path)
    # loop over batches
    for batch in ohsu:
        # parse xml
        root = ETree.fromstring(batch)
        # loop through each doc in the batch
        for doc in tqdm(root):
            docno = ''
            body = ''
            # loop through each element (tag, value)
            for elem in doc:
                if elem.tag == 'DOCNO':
                    docno = elem.text.strip()
                else:
                    body = elem.text.strip()
            # return doc to index
            yield docno, body


def read_ohsu_queries(query_path):
    """read query file and return a dict[id] = {title: <string>, desc: <string>}"""
    with open(query_path, 'r') as qf:
        q = qf.read()
    q = [query.split('\n') for query in q.split('\n\n') if query]
    # loop through each query and fill dict
    qdict = dict()
    for query in q:
        qid = query[1].split()[-1]
        qdict[qid] = dict()
        qdict[qid]['title'] = query[2].split('<title>')[1].strip()
        qdict[qid]['desc'] = query[4]
    return qdict


def read_cds_queries(query_path):
    """read query file and return a dict[id] = {note: <string>, description: <string>, summary: <string>}"""
    with open(query_path, 'r') as qf:
        queries = qf.read()
    # convert queries to xml
    q = ETree.fromstring(queries)
    # loop through each query and fill dict
    qdict = dict()
    for query in q:
        qid = query.attrib['number']
        qdict[qid] = dict()
        # store query versions (i.e. note, description, summary)
        for version in query:
            qdict[qid][version.tag] = version.text.strip()
    return qdict


def get_score(run, qrels, measure):
    """return np array of scores for a given measure"""
    if "P_" in measure:
        cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + " " + qrels + " " + run
    elif "ndcg_cut" in measure:
        cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
    else:
        cmd = "./trec_eval/trec_eval -q -m " + measure + " " + qrels + " " + run
    # run trev_eval as a subprocess
    process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    result = process.stdout.decode('utf-8').split('\n')
    # get scores
    scores = np.array([(score.split('\t')[-2], score.split('\t')[-1]) for score in result 
        if score.split('\t')[0].strip() == measure and score.split('\t')[-2] != 'all'])
    return scores


def get_averaged_measure_score(run, qrels, measure):
    """return averaged measure score over topics"""
    if "P_" in measure:
        cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + " " + qrels + " " + run
    elif "ndcg_cut" in measure:
        cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
    else:
        cmd = "./trec_eval/trec_eval -m " + measure + " " + qrels + " " + run
    process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    result = process.stdout.decode('utf-8').split('\n')
    if 'recall.' in measure:
        measure = '_'.join(measure.split('.'))
    qscore = np.array([score.split('\t')[-1] for score in result if score.split('\t')[0].strip() == measure])
    qscore = qscore.astype(np.float)[0]
    return qscore


def evaluate(measures, ranking_folder, ranking_name, qrels_folder, qrels_name):
    """evaluate models on given measures"""
    scores = []
    print('evaluate model ranking')
    if type(measures) == list:  # list of measures provided
        for measure in measures:
            scores.append(get_averaged_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measure)) 
    else:  # single measure provided
        scores.append(get_averaged_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measures))
    print("\t".join([measure + ": " + str(score) for measure, score in zip(measures, scores)]))
    return [(measure, score) for measure, score in zip(measures, scores)]
