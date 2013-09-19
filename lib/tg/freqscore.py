"""
translation scores based on frequency
"""

import logging 
import cPickle

from tg.scorer import Scorer


# TODO: 
# - unit test
# - no counts for multi-words
# - how to handle weight of MWU relative to single words?
# - lempos-based counts - requiter another pos mapping step



log = logging.getLogger(__name__)


def wrap(graph, *args, **kwargs):
    print graph
    

class FreqScorer(Scorer):
    """
    score translation candidates according to their frequency
    """
    
    def __init__(self, counts_pkl_fname, score_attr="freq_score", oov_count=0):
        Scorer.__init__(self, score_attr)
        self.counts_pkl_fname = counts_pkl_fname
        self.oov_count = oov_count
        
    def __call__(self, obj, *args, **kwargs):      
        log.info("reading counts from " + self.counts_pkl_fname)
        self.counts_dict = cPickle.load(open(self.counts_pkl_fname))  
        Scorer.__call__(self, obj, *args, **kwargs)   

    def _score_translations(self, graph, u):
        edge_data = []
        counts = []
        
        for u, v, data in graph.trans_edges_iter(u):
            edge_data.append(data)
            lemma = graph.lemma(v)
            # if oov_count is not None, all translation edges will have a
            # freq_score attrib
            count = self.counts_dict.get(lemma, self.oov_count)
            log.debug(u"lemma '{}' has count {}".format(lemma, count))
            counts.append(count)
            
        return edge_data, counts
