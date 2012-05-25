"""
translation scores based on frequency
"""

import logging 
import cPickle

import graphproc


# TODO: 
# - unit test
# - no counts for multi-words
# - how to handle weight of MWU relative to single words?
# - lempos-based counts - requiter another pos mapping step



log = logging.getLogger(__name__)


def wrap(graph, *args, **kwargs):
    print graph
    

class FreqScore(graphproc.GraphProces):
    """
    score translation candidates according to their frequency
    """
    
    def __init__(self, counts_pkl_fname, score_attr="freq_score"):
        log.info("reading counts from " + counts_pkl_fname)
        self.counts_dict = cPickle.load(open(counts_pkl_fname))
        self.score_attr = score_attr
        self.oov_count = 0    

    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        for u in graph.source_nodes_iter():
            edge_data = []
            edge_counts = []
            total = 0.0
            best_count = None
            
            for u,v,data in graph.trans_edges_iter(u):
                count = self.count(graph, v)
                edge_counts.append(count)
                total += count
                edge_data.append(data)
                best_count = max(count, best_count)
            
            # convert counts to probabilities
            for count, data in zip(edge_counts, edge_data):
                try:
                    data[self.score_attr] = count / total 
                except ZeroDivisionError:
                    data[self.score_attr] = 0.0
                
    def count(self, graph, v):
        lemma = graph.lemma(v)
        count = self.counts_dict.get(lemma, self.oov_count)
        log.debug(u"lemma '{0}' has count {1}".format(lemma, count))
        return count
