"""
scoring with naive bayes models
"""

import logging

import numpy as np
import scipy.sparse as sp

import sklearn.naive_bayes as nb

import graphproc


log = logging.getLogger(__name__)


class NBScore(graphproc.GraphProces):
    """
    add Naive Bayes classifier scores to translation candidates
    """
    
    def __init__(self, vocab, model, score_attr="nb_score"):
        self.vocab = vocab 
        self.model = model
        self.score_attr = score_attr
        self.classifier = nb.BernoulliNB()
    
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        lemma_vectors = self._make_matrix(graph)
        # result of summation is dense vector
        sent_vec = sp.csr_matrix(lemma_vectors.sum(axis=0))
        
        for u, lemma_vec in zip(graph.source_nodes_iter(), lemma_vectors):
            # subtract translation candidates for current source node
            # from sentence context vector
            context_vec = sent_vec - lemma_vec

            try:
                source_lempos = " ".join(graph.node[u]["lex_lempos"])
            except KeyError:
                # not found in lexicon, so no model available
                continue

            try:
                model_data = self.model["/models/" + source_lempos]
            except KeyError:
                # no model available for lempos
                continue

            print "***", source_lempos                
            self.classifier.class_log_prior_ = model_data["class_log_prior"]
            self.classifier.feature_log_prob_ = model_data["feature_log_prob"]
            target_names = model_data["target_names"]
            self.classifier.unique_y = np.arange(len(target_names))            
            preds = self.classifier.predict_proba(context_vec)
            lempos2prob = dict(zip(target_names, preds[0]))
            print lempos2prob
            
            for u,v,data in graph.trans_edges_iter(u):
                # TODO: handle source/target hypernodes 
                if graph.is_target_node(v):
                    # FIX: replace is unnecessary
                    target_lempos = graph.lempos(v).replace("/", "_")
                    try:
                        data[self.score_attr] = lempos2prob[target_lempos]
                    except KeyError:
                        # model does not predict this target lemma
                        continue
                    
    def _make_matrix(self, graph):
        dim = (len(graph), len(self.vocab))
        mat = sp.lil_matrix(dim, dtype=np.int16)
        
        for row_i, u in enumerate(graph.source_nodes_iter()):
            for u,v,data in graph.trans_edges_iter(u):
                # TODO: handle source/target hypernodes 
                if graph.is_target_node(v):
                    target_lemma = graph.node[v]["lemma"]
                    try:
                        col_j = self.vocab[target_lemma]
                    except KeyError:
                        # oov
                        continue
                    mat[row_i, col_j] += 1
                    
        mat = mat.tocsr()
        # remove superfluous rows now that no of source nodes is known
        return mat[:row_i, :]
    
