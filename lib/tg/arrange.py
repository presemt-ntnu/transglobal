"""
arrange best translations into target language expression 
"""

import graphproc


class Arrange(graphproc.GraphProces):
    
    def _single_run(self, graph):
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
            
        return " ".join(target_lemmas)



if __name__ == "__main__":
    import cPickle
    
    from draw import Draw
    
    graph_list = cPickle.load(open("graphs-freq.pkl"))    

    arranger = Arrange() 
    target_sentences = arranger(graph_list)
    
    print "\n".join(target_sentences).encode("utf8")
    
    