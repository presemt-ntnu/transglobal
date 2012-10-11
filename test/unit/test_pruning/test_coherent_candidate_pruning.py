import cPickle
import unittest
from tg.pruning.coherent_candidate_pruning import CoherentCandidatePruner

def _highest_cset_entropy(graph):
    return CoherentCandidatePruner._get_entropy(graph, CoherentCandidatePruner._get_highest_ent_source_node(graph))

class TestCoherentCandidatePruning(unittest.TestCase):
    graphs_fn = 'test_pruning/data/graphs_en-de_centroid.pkl'

    def setUp(self):
        self.graphs = cPickle.load(open(self.graphs_fn))

    def test_fixed_cutoff(self):
        cutoffs = [1, 1.5, 2, 2.5]

        for c in cutoffs:
            pruner = CoherentCandidatePruner(cutoff = c)
            pruned_graphs = pruner(self.graphs, copy = True)

            for g in pruned_graphs:
                assert _highest_cset_entropy(g) < c

    def test_mge_cutoff(self):
        pruner = CoherentCandidatePruner(cutoff = 'graph_mean_entropy')
        pruned_graphs = pruner(self.graphs, copy = True)

        for p_g, g in zip(pruned_graphs, self.graphs):
            assert _highest_cset_entropy(p_g) < pruner._mean_graph_entropy(g)
