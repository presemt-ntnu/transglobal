"""
approximate maximum score  
"""

import logging 
import xml.etree.cElementTree as et
from collections import Counter

from tg.graphproc import GraphProcess
from tg.mteval import read_ref_trans_counts

log = logging.getLogger(__name__)


# TODO
# - counts for MWUs can be obtained by counting n-grams in the references
# - store doc-id and seg-id during annotation?


class DictUpperScore(GraphProcess):
    """
    Approximation of the maximal scoring obtainable.
    
    For each segment (sentence), we look at its reference translation(s) and
    count the number of times that a target lemma occurs. Next, when scoring
    translations candidates, we pick the translations with the highest count
    in the reference translation.
    
    Remarks:
    - This is an approximation. It may not work well for high frequency words 
      such as determiners or pronouns.
    - It is assumed that the order of documents (by docid) in the source
      and reference is the same    
    - In case of ties (candidates with equal counts), the choice is arbitrary.
    - In case all candidates have zero conts, the choice is arbitary
      (does not matter for score anyway).
    - Multi-word units are not taken into account yet.
    """
    
    def __init__(self, ref_fname, score_attr="dup_score"):
        GraphProcess.__init__(self)
        self.score_attr = score_attr
        # It is assumed that the order of documents (by docid) in the source
        # and reference is the same
        self.counts = read_ref_trans_counts(ref_fname, flatten=True)
    
    def _batch_run(self, obj_list, *args, **kwargs):
        self.seg_num = 0
        GraphProcess._batch_run(self, obj_list, *args, **kwargs)

                
    def _single_run(self, graph):
        log.debug("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        seg_counts = self.counts[self.seg_num]
        
        for u in graph.source_nodes_iter():
            edge_data = []
            counts = []
            
            for u,v,data in graph.trans_edges_iter(u):
                edge_data.append(data)
                target_lemma = graph.lemma(v).lower()
                lemma_count = seg_counts.get(target_lemma, 0)
                counts.append(lemma_count)
                    
            # normalize scores        
            total = float(sum(counts))
            
            for data, count in zip(edge_data, counts):
                try:
                    data[self.score_attr] = count / total
                except ZeroDivisionError:
                    data[self.score_attr] = 0.0
                
        self.seg_num += 1