"""
scoring with naive bayes models
"""

import logging

import numpy as np
import operator
import scipy.sparse as sp


from sklearn.naive_bayes import MultinomialNB

from tg.graphproc import GraphProces


log = logging.getLogger(__name__)


class NBScore(GraphProces):
    """
    add Naive Bayes classifier scores to translation candidates
    """
    
    def __init__(self, classifier, score_attr="nb_score"):
        self.classifier = classifier
        self.score_attr = score_attr
        
        if log.isEnabledFor(logging.debug):
            self.reversed_vocab = zip(*sorted(self.classifier.vocab.items(), key=operator.itemgetter(1)))[0]
    
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        lemma_vectors = self._make_matrix(graph)
        # result of summation is dense vector
        sent_vec = sp.csr_matrix(lemma_vectors.sum(axis=0))
        
        if log.isEnabledFor(logging.debug):
            log.debug("source sentence lemmas: " + " ".join(graph.source_lemmas()))
            sent_items = [ u"{0}={1}".format(self.reversed_vocab[i], count)
                      for i, count in zip(sent_vec.indices, sent_vec.data) ]
            items_str = u", ".join(sent_items)
            log.debug(u"sent_vec: {0}\n".format(items_str))
        
        for u, lemma_vec in zip(graph.source_nodes_iter(ordered=True), lemma_vectors):
            try:
                source_lempos = " ".join(graph.node[u]["lex_lempos"])
            except KeyError:
                log.debug(u"(annotated) source lempos {0} not in lexicon\n".format(graph.lempos(u)))
                # not found in lexicon, so no model available
                continue
            
            # subtract translation candidates for current source node
            # from sentence context vector
            context_vec = sent_vec - lemma_vec
            lempos2score = self.classifier.score(source_lempos, context_vec)
            
            if not lempos2score:
                log.debug(u"no model available for (lexicon) source lempos {0}\n".format(source_lempos))
                # no model for source lempos combination
                continue
            
            if log.isEnabledFor(logging.debug):
                lemma_items = [ u"{0}={1}".format(self.reversed_vocab[i], count)
                                  for i, count in zip(lemma_vec.indices, lemma_vec.data) ]
                lemma_str = u", ".join(lemma_items)
                log.debug(u"lemma_vec for lempos {0}: {1}".format(source_lempos, lemma_str))
                
                context_items = [ u"{0}={1}".format(self.reversed_vocab[i], count)
                                  for i, count in zip(context_vec.indices, context_vec.data) ]
                context_str = u", ".join(context_items)
                log.debug(u"context_vec for lempos {0}: {1}".format(source_lempos, context_str))
                
                score_items = sorted(lempos2score.items(), reverse=True, key=operator.itemgetter(1))
                score_str = u", ".join(u"{0}={1:.4f}".format(*pair) for pair in score_items)
                log.debug(u"scores for lempos {0}: {1}\n".format(source_lempos, score_str))
                
            best_score, best_node = None, None
            
            for u,v,data in graph.trans_edges_iter(u):
                # TODO: handle source/target hypernodes 
                if graph.is_target_node(v):
                    target_lempos = graph.lempos(v)
                    
                    try:
                        score = data[self.score_attr] = lempos2score[target_lempos]
                    except KeyError:
                        # model does not predict this target lemma,
                        # (which may be different from a 0.0 score)
                        continue
                    
                    if score > best_score:
                        best_score, best_node = score, v

            graph.node[u].setdefault("best_nodes", {})[self.score_attr] = best_node
                
        
                    
    def _make_matrix(self, graph):
        dim = (len(graph), len(self.classifier.vocab))
        mat = sp.lil_matrix(dim, dtype=np.int16)
        
        for row_i, u in enumerate(graph.source_nodes_iter(ordered=True)):
            for u,v,data in graph.trans_edges_iter(u):
                # TODO: handle source/target hypernodes 
                if graph.is_target_node(v):
                    target_lemma = graph.node[v]["lemma"]
                    try:
                        col_j = self.classifier.vocab[target_lemma]
                    except KeyError:
                        # oov
                        continue
                    mat[row_i, col_j] += 1
                    
        mat = mat.tocsr()
        # remove superfluous rows now that no of source nodes is known
        return mat[:row_i, :]
    
