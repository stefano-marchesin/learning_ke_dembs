# When Reproducibility goes Sideways: The Case of Knowledge-Enhanced Document Embeddings

Repository to contain code and information for the paper: 
<p align="center">
<b><i>When Reproducibility goes Sideways: The Case of Knowledge-Enhanced Document Embeddings</i></b>
 </p>
Submitted to ECIR 2020 reproducibility track by S. Marchesin, G. Silvello, and M. Agosti

### Requirements

- ElasticSearch 6.6
- Python 3
  - Numpy
  - TensorFlow >= 1.13
  - Whoosh
  - SQLite3
  - Cvangysel
  - Pytrec_Eval
  - Scikit-Learn
  - Tqdm
  - QuickUMLS
  - Elasticsearch
  - Elasticsearch_dsl
- UMLS 2018AA

### Additional Notes
``server.py`` needs to be substitued within QuickUMLS folder as it contains a modified version required to run knowledge-enhanced models.  
The folder structure required to run experiments can be seen in folder ``example``. Python files need to be put in root.  
Qrels file needs to be in ``.txt`` format.  
To perform retrofitting run ``retrofit_doc_vecs.py``, whereas to train PV-DM and cDoc2Vec models run ``gensim_doc2vec.py``.  
To run BM25, use the Jupyter Notebook file ``elastic_search.ipynb``.  
To perform query expansion run ``qe_combsum.py``.
