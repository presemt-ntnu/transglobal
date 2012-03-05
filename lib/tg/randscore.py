"""
random translation probabilities
"""

import random

import graphproc


class RandProb(graphproc.GraphProces):
    """
    add random probabilities to translation candidates
    """
    
    def _single_run(self, graph):
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
        #return 10


if __name__ == "__main__":
    import cPickle
    
    from draw import Draw
    
    graph_list = cPickle.load(open("graphs.pkl"))
    g = graph_list[3]
    
    d = RandProb()
    d(g)
    
    draw = Draw(g)
    draw.write("g3-random.pdf", format="pdf")

