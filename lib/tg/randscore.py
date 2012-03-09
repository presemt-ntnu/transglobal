"""
random translation probabilities
"""

import logging
import random

import graphproc


log = logging.getLogger(__name__)


class RandProb(graphproc.GraphProces):
    """
    add random probabilities to translation candidates
    """
    
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        for u in graph.source_nodes_iter():
            edge_data = []
            edge_counts = []
            total = 0.0
                        
            for u,v,data in graph.trans_edges_iter(u):
                count = self.count(graph, v)
                edge_counts.append(count)
                total += count
                edge_data.append(data)
                    
            for count, data in zip(edge_counts, edge_data):
                data["score"] = count / total 
                
    def count(self, graph, v):
        return random.randint(1,1000)   

