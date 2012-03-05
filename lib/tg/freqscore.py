"""
translation scores based on frequency
"""

import cPickle

import graphproc


class FreqWeight(graphproc.GraphProces):
    """
    score translation candidates according to their frequency
    """
    
    def __init__(self, counts_pkl_fname):
        self.counts_dict = cPickle.load(open(counts_pkl_fname))
        self.oov_count = 0
    
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
                try:
                    data["score"] = count / total 
                except ZeroDivisionError:
                    data["score"] = 0.0
                
    def count(self, graph, v):
        # TODO: handle hyper nodes
        if graph.is_hyper_node(v):
            return self.oov_count
        
        lemma = graph.node[v]["lemma"]
        return self.counts_dict.get(lemma, self.oov_count)
        


if __name__ == "__main__":
    import cPickle
    
    from draw import Draw
    
    graph_list = cPickle.load(open("graphs.pkl"))
    
    d = FreqWeight("../../data/freqs/deTenTen-lemma-count-cutoff-10.pkl")
    
    d(graph_list)
    
for i, graph in enumerate(graph_list):
    draw = Draw(graph)
    draw.write("g{0}-freq.pdf".format(i), format="pdf")


# save
cPickle.dump(graph_list, open("graphs-freq.pkl", "wb"))