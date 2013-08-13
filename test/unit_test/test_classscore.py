"""
unit test for classscore
"""

import cPickle

from tg.config import config
from tg.classcore import ClassifierScore
from tg.classify import TranslationClassifier
from tg.format import TextFormat


class TestClassifierScore:
    
    """
    Note: these tests focus on the lempos "absolut/adj". This is contained in
    data/de-en_ambig.tab. Therefore there is a disambiguation model for it in
    data/de-en_models-hdf5_. It also ocurs as the 3rd source word in the
    second sentence/graph of data/graphs_sample_out_de-en.pkl.
    """
    
    @classmethod
    def setup_class(cls):
        models_hdf_fname = config["test_data_dir"] + "/de-en_models.hdf5_"
        # make a translation classifier that uses this model
        cls.classifier = TranslationClassifier(models_hdf_fname)
        
    def test_classifier_score_full(self):
        self._classifier_score("full")
        
    def test_classifier_score_mft(self):
        self._classifier_score("mft")
    
    def _classifier_score(self, vectorizer):
        # make a scorer that uses this classifier
        class_score = ClassifierScore(self.classifier, vectorizer=vectorizer)
        
        graph = cPickle.load(
            open(config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"))[1]
        class_score(graph)
        
        # find source node for "absolut/adj"
        for sn in graph.source_nodes_iter():
            if graph.node[sn]["lex_lempos"] == ["absolut/adj"]:
                break
        
        # target lemmas included in the model for "abolut/adj"
        target_lemmas = 'total', 'absolute', 'thorough'
        # edge attribute containing classifier score
        score_attr = "class_score"

        # make sure that all and only edges to targeted lemmas
        # have a classifier score attribute 
        for _, tn, e in graph.trans_edges_iter(sn):
            if graph.lemma(tn) in target_lemmas:
                assert e.has_key(score_attr)
            else:
                assert not e.has_key(score_attr)
        
        
        
        
if __name__ == "__main__":
    #from tg.utils import set_default_log
    #set_default_log()
    import nose
    nose.main(defaultTest=__name__)
