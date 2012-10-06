from tg.graphproc import GraphProcess
import scipy.stats as stats
from operator import itemgetter
import numpy as np
import math

GRAPH_MEAN_ENTROPY = 'graph_mean_entropy'

class CoherentCandidatePruner(GraphProcess):
    """
    Prunes the centroid probabilities of translation edges by iteratively removing
    low probabilities from source node edges with high entropy over outgoing
    translation edges.

    Parameters
    ----------
    cutoff: Stop pruning when all translation edge centroid distributions have an entropy
        lower than the cutoff. The constant GRAPH_MEAN_ENTROPY specifies the cutoff as the mean
        of the initial entropies.

    Notes
    -----
    Use generic __call__ method on parent `GraphProc`.
    """
    def __init__(self, cutoff = GRAPH_MEAN_ENTROPY):
        self.cutoff = cutoff

    def _single_run(self, graph, copy = False, *args, **kwargs):
        """
        @param copy: If set to True a deep copy of the passed graph instance
            will be pruned and returned.
        """
        if copy:
            graph = graph.copy()

        cutoff = self._get_cutoff(graph)

        while True:
            source_node = self._get_highest_ent_source_node(graph)
            highest_entropy = self._get_entropy(graph, source_node)

            if highest_entropy <= cutoff:
                break
            self._prune_source_node(graph, source_node)

        return graph

    def _get_cutoff(self, graph):
        if self.cutoff == GRAPH_MEAN_ENTROPY:
            return self._mean_graph_entropy(graph)
        else:
            return self.cutoff

    @staticmethod
    def _get_entropy(graph, source_node):
        probs = []

        for _, tn, meta in graph.trans_edges_iter(source_node):
            if meta.has_key('centroid_score') and meta['centroid_score'] > 0:
                probs.append(meta['centroid_score'])

        if len(probs) > 0:
            return stats.entropy(probs)
        else:
            return float('nan')

    @staticmethod
    def _get_entropies(graph):
        entropies = []

        for sn in graph.source_nodes():
            entropy = CoherentCandidatePruner._get_entropy(graph, sn)

            if not math.isnan(entropy):
                entropies.append((sn, entropy))

        return entropies

    @staticmethod
    def _get_highest_ent_source_node(graph):
        sorted_entropies = sorted(CoherentCandidatePruner._get_entropies(graph),
            key = itemgetter(1), reverse = True)

        if len(sorted_entropies) < 1:
            raise ValueError

        return sorted_entropies[0][0]

    @staticmethod
    def _get_translation_scores(graph, source_node):
        scores = []

        for _, tn, meta in graph.trans_edges_iter(source_node):
            if meta.has_key('centroid_score') and meta['centroid_score'] > 0.0:
                scores.append((tn, meta['centroid_score']))

        return scores

    @staticmethod
    def _reweight_scores(graph, source_node):
        scores = [(tn, meta['centroid_score']) for _, tn, meta
                  in graph.trans_edges_iter(source_node) if meta.has_key('centroid_score')]

        total_score = float(sum([score for _, score in scores]))

        for tn, score in scores:
            graph.edge[source_node][tn]['centroid_score'] = score / total_score

    @staticmethod
    def _prune_source_node(graph, source_node):
        sorted_scores = sorted(CoherentCandidatePruner._get_translation_scores(graph, source_node),
            key = itemgetter(1))
        target_node = sorted_scores[0][0]

        graph.edge[source_node][target_node]['centroid_score'] = 0.0
        CoherentCandidatePruner._reweight_scores(graph, source_node)

    @staticmethod
    def _mean_graph_entropy(graph):
        entropies = CoherentCandidatePruner._get_entropies(graph)

        return np.mean([ent for (_, ent) in entropies])
