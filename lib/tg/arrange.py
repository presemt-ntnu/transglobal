"""
arrange best translations into target language expressions 
"""

# TODO: handle hypernodes

import logging

import graphproc


log = logging.getLogger(__name__)


class Arrange(graphproc.GraphProces):
    
    def __init__(self, score_attrs=["freq_score"], *args, **kwargs):
        graphproc.GraphProces.__init__(self, *args, **kwargs)
        self.score_attrs = score_attrs
    
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        target_lemmas = []
        
        for u, data in graph.source_nodes_iter(ordered=True, data=True):
            for score_attr in self.score_attrs:
                try:
                    best_node = data["best_nodes"][score_attr]
                except KeyError:
                    continue
                else:
                    # TODO: handle hypernodes
                    best_lemma = graph.node[best_node].get("lemma", "__UNKNOWN__")   
                    target_lemmas.append(best_lemma)   
                    graph.node[best_node]["best"] = True
                    break
            
        graph.graph["target_lemma"] = target_lemmas
        graph.graph["target_string"] = u" ".join(target_lemmas)
        
        
    def score(self, data):
        return data.get(self.score_attr, -1)
        
        
        

