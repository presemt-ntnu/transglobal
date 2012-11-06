import unittest
import cPickle
from tg.interpolated_score import InterpolatedScore
from tg.config import config

class TestInterpolatedScore(unittest.TestCase):
    graphs_fn = config["test_data_dir"] + '/graphs_en-de_freq-centroid.pkl'

    def setUp(self):
        self.graphs = cPickle.load(open(self.graphs_fn))

    def test_even_centroid_freq_scoring(self):
        scorer = InterpolatedScore(score_attrs = ['freq_score', 'centroid_score'], weights = 'even')
        scored_graphs = scorer(self.graphs, copy = True)

        for g, sg in zip(self.graphs, scored_graphs):
            for sn, tn, meta in g.trans_edges_iter():
                if meta.has_key('freq_score') and meta.has_key('centroid_score'):
                    for _, s_tn, s_meta in sg.trans_edges_iter(sn):
                        if s_tn == tn:
                            assert s_meta[scorer.score_attr] == 0.5 * meta['centroid_score'] + 0.5 * meta['freq_score']
                            break

    def test_skewed_centroid_freq_scoring(self):
        scorer = InterpolatedScore(score_attrs = ['freq_score', 'centroid_score'], weights = [0.2, 0.8])
        scored_graphs = scorer(self.graphs, copy = True)

        for g, sg in zip(self.graphs, scored_graphs):
            for sn, tn, meta in g.trans_edges_iter():
                if meta.has_key('freq_score') and meta.has_key('centroid_score'):
                    for _, s_tn, s_meta in sg.trans_edges_iter(sn):
                        if s_tn == tn:
                            assert s_meta[scorer.score_attr] == 0.8 * meta['centroid_score'] + 0.2 * meta['freq_score']
                            break