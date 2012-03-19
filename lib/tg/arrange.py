"""
arrange best translations into target language expressions 
"""

# TODO: handle hypernodes

import logging

import graphproc


log = logging.getLogger(__name__)


class Arrange(graphproc.GraphProces):
    
    def _single_run(self, graph):
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        target_lemmas = []
        
        for u in graph.source_nodes_iter(ordered=True):
            best_score = -1
            best_lemma = ""
            
            for u,v,data in graph.trans_edges_iter(u):
                score = data.get("score", -1)
                if score > best_score:
                    best_score = score
                    # TODO: handle hypernodes
                    best_lemma = graph.node[v].get("lemma", "XXX")
                    
            target_lemmas.append(best_lemma)
            
        graph.graph["target_lemma"] = target_lemmas
        graph.graph["target_string"] = u" ".join(target_lemmas)
        
        

