"""
scoring with naive bayes models
"""

import logging

import numpy as np
import scipy.sparse as sp

from sklearn.naive_bayes import MultinomialNB

from tg.graphproc import GraphProces


log = logging.getLogger(__name__)


class NBScore(GraphProces):
    """
    add Naive Bayes classifier scores to translation candidates
    """
    
    def __init__(self, vocab, classifier, score_attr="nb_score"):
        self.vocab = vocab 
        self.classifier = classifier
        self.score_attr = score_attr
    
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        lemma_vectors = self._make_matrix(graph)
        # result of summation is dense vector
        sent_vec = sp.csr_matrix(lemma_vectors.sum(axis=0))
        
        for u, lemma_vec in zip(graph.source_nodes_iter(), lemma_vectors):
            try:
                source_lempos = " ".join(graph.node[u]["lex_lempos"])
            except KeyError:
                # not found in lexicon, so no model available
                continue
            
            # subtract translation candidates for current source node
            # from sentence context vector
            context_vec = sent_vec - lemma_vec
            lempos2score = self.classifier.score(source_lempos, context_vec)
            
            if not lempos2score:
                # no model for source lempos combination
                continue
            
            for u,v,data in graph.trans_edges_iter(u):
                # TODO: handle source/target hypernodes 
                if graph.is_target_node(v):
                    target_lempos = graph.lempos(v)
                    
                    try:
                        data[self.score_attr] = lempos2score[target_lempos]
                    except KeyError:
                        # model does not predict this target lemma,
                        # (which may be different from a 0.0 score)
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
    
