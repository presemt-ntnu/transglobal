from graphproc import GraphProcess
import numpy as np

EVEN_WEIGHTS = 'even'

class InterpolatedScore(GraphProcess):
    """
    Combines translation edge scores into weighted score.
    """

    def __init__(self, score_attrs = ['freq_score'], weights = EVEN_WEIGHTS, score_attr = 'weighted_score'):
        """
        @param score_attrs: List of attribute identifiers to interpolate into score.
        @param weights: Interpolation weights for attributes in the same order as given in the
            attribute list.
        @param score_attr: Attribute where weighted score is stored.
        """

        self.score_attrs = score_attrs
        self.score_attr = score_attr
        self.weights = self._get_weights(weights, score_attrs)

    def _single_run(self, graph, copy = False, *args, **kwargs):
        """
        @param copy: Score and return a deep copy of the passed graph instance.
        """

        if copy:
            graph = graph.copy()

        for _, _, meta in graph.trans_edges_iter():
            try:
                scores = [meta[attr] for attr in self.score_attrs]
                meta[self.score_attr] = np.dot(self.weights, np.array(scores))
            except KeyError:
                continue

        return graph

    @staticmethod
    def _get_weights(weights, score_attrs):
        if weights == EVEN_WEIGHTS:
            return np.ones(len(score_attrs)) / len(score_attrs)
        else:
            return np.array(weights)
