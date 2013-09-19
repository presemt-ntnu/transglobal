"""
random translation scores
"""

import numpy as np

from tg.scorer import Scorer


class RandScorer(Scorer):
    """
    add random scores to translation candidates
    """
    
    def __init__(self, score_attr="rand_score"):
        Scorer.__init__(self, score_attr)
                
    def _score_translations(self, graph, u):
        edge_data = [data for u,v,data in graph.trans_edges_iter(u)]
        scores = np.random.random(len(edge_data))
        return edge_data, scores

