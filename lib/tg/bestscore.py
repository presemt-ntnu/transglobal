"""
combine scores into best score
"""

# TODO: handle hypernodes

import logging

import graphproc


log = logging.getLogger(__name__)


class BestScore(graphproc.GraphProces):
    
    def __init__(self, base_score_attrs, score_attr="best_score", *args, **kwargs):
        graphproc.GraphProces.__init__(self, *args, **kwargs)
        self.base_score_attrs = base_score_attrs
        self.best_score_attr = score_attr
        
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        # TODO: handle hypernodes        
        for u in graph.source_nodes_iter(ordered=True):
            score_seen = False
            
            for score_attr in self.base_score_attrs:
                for u, v, d in graph.trans_edges_iter(u):
                    try:
                        d[self.best_score_attr] = d[score_attr]
                    except KeyError:
                        continue
                    else:
                        score_seen = True
                if score_seen:
                    break
        