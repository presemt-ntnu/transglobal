"""
base class for scorers
"""

from tg.graphproc import GraphProcess


class Scorer(GraphProcess):
    
    def __init__(self, score_attr):
        GraphProcess.__init__(self)
        self.score_attr = score_attr
        
    def _single_run(self, graph):
        GraphProcess._single_run(self, graph)
        
        # Iterate over source nodes rather than directly over translation
        # edges, because scores need to be normalized per source node.
        for u in graph.source_nodes_iter():
            self._add_normalized_scores(
                *self._score_translations(graph, u))
            
    def _score_translations(self, graph, u):
        """
        Compute raw scores for the translation candidates for source node u
        
        Parameters
        ----------
        graph: TransGraph
            translation graph
        u: str
            source node identifier
            
        Returns
        -------
        edge_data, scores: list of dicts, list of floats
            list of data associated with u's translation edges
            list of scores
            
        Note
        ----
        Must be implemented by subclass
        """
        return [], []
    
    def _add_normalized_scores(self, edge_data, scores):
        """
        Add normalized scores to translation edges 
        
        Parameters
        ----------
        edge_data: dict
            data dict for edges
        scores: iterable
            scores
        """
        total = float(sum(scores))
        
        for data, score in zip(edge_data, scores):
            try:
                data[self.score_attr] = score / total
            except ZeroDivisionError:
                data[self.score_attr] = 0.0