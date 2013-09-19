"""
combine scores into best score
"""

# TODO: handle hypernodes

import logging

from tg.graphproc import GraphProcess
from tg.scorer import Scorer


log = logging.getLogger(__name__)


class BestScorer(Scorer):
    """
    BestScorer combines scores from multiple scorer according to their
    priority. Suppose base_score_attrs=['nb_score', 'freq_score']. If any of
    translation edges of a source node contains the 'nb_score' attribute,
    then the value of their 'best_score' attribute becomes that of the
    'nb_score' attribute (or zero for those that have no 'nb_score'
    attribute). However, if none of the translation edges has the 'nb_score'
    attribute, then the same procedure is repeated for 'freq_score', and so
    on for all attributes in base_score_attrs. 
    """
    
    def __init__(self, base_score_attrs, score_attr="best_score"):
        Scorer.__init__(self, score_attr)
        self.base_score_attrs = base_score_attrs
        
    def _single_run(self, graph):
        # TODO: handle hypernodes  
        GraphProcess._single_run(self, graph)
        
        # skip normalization, because base scores are already normalized
        for u in graph.source_nodes_iter():
            # first pass to figure out which score attr is present
            base_score_attr = self._find_base_score_attr(graph, u)  
            # second pass to add the best scores
            for u, v, data in graph.trans_edges_iter(u):
                data[self.score_attr] = data.get(base_score_attr, 0.0)
                
    def _find_base_score_attr(self, graph, u):
        has_trans_edge = False
        
        for score_attr in self.base_score_attrs:
            for u, v, data in graph.trans_edges_iter(u):
                has_trans_edge = True
                if data.has_key(score_attr):
                    return score_attr
        
        # skip warning if source node has no translations, e.g. punctuation
        if has_trans_edge:
            log.warning("none of the translation edges of node {} in graph {} "
                        "has any of the base score attributes {}".format(
                            u, graph, self.base_score_attrs))

