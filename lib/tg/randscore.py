"""
random translation probabilities
"""

import logging
#import random

import numpy as np

import graphproc


log = logging.getLogger(__name__)


class RandProb(graphproc.GraphProcess):
    """
    add random probabilities to translation candidates
    """
    
    def __init__(self, score_attr="rand_score"):
        self.score_attr = score_attr
                
    def _single_run(self, graph):
        log.debug("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        for u in graph.source_nodes_iter():
            edge_data = [data for u,v,data in graph.trans_edges_iter(u)]
            rand_scores = np.random.random(len(edge_data))
            rand_scores = rand_scores / sum(rand_scores)
            for data, score in zip(edge_data, rand_scores):
                data[self.score_attr] = score 

