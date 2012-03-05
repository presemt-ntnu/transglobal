

class Disambiguate:
    
    def single_run(self, graph):
        translation = []
        
        for u in graph.source_nodes_iter():
            best_lemma = ""
            best_score = 0
            
            for u,v,data in graph.out_edges_iter(u, data=True):
                score = data.get("score", 0)
                
                if score > best_score:
                    best_score = score
                    best_lemma = self.get_target_lemmas(v)
                
            translation.append(best_lemma)
        
        return " ".join(translation)
    
                
                
import cPickle

from draw import Draw

graph_list = cPickle.load(open("graphs.pkl"))
g = graph_list[3]

d = Disambiguate()
d.single_run(g)

draw = Draw(g)
draw.write("g3-scores.pdf", format="pdf")

