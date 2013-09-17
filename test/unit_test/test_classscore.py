"""
unit test for classscore
"""

import cPickle

import h5py

from tg.config import config
from tg.classcore import ClassifierScore, Vectorizer
from tg.classify import TranslationClassifier
from tg.format import TextFormat



class TestVectorizer:
    
    @classmethod
    def setup_class(cls):
        # get vocab
        samp_fname = config["test_data_dir"] + "/de-en_samples.hdf5_"
        fh = h5py.File(samp_fname, "r")
        utf8_vocab = fh["vocab"][()] 
        cls.vocab = dict((lemma.decode("utf-8"), i) 
                         for i,lemma in enumerate(utf8_vocab))
        # get graph
        graphs_fname = config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"
        cls.graph = cPickle.load(open(graphs_fname))[0]
        
    def test_full_vectors(self):
        vectorizer = Vectorizer()
        m = vectorizer(self.graph, self.vocab)
        # 1: the --> who, the, which, that
        # NB none of these translations are in the vocabulary
        assert m[0].nnz == 0
        # 2: europaische --> European, continental
        assert m[1].nnz == 2
        assert m[1, self.vocab["European"]] == 1.0
        assert m[1, self.vocab["continental"]] == 1.0
        # 3: Union --> union
        assert m[2].nnz == 1
        assert m[2, self.vocab["union"]] == 1.0
        # 4: hat --> own, have, bear, entertain, experience, gotta
        # NB "have" and "own" are not in vocab
        assert m[3].nnz == 4
        assert m[3, self.vocab["bear"]] == 1.0
        assert m[3, self.vocab["entertain"]] == 1.0
        assert m[3, self.vocab["experience"]] == 1.0
        assert m[3, self.vocab["gotta"]] == 1.0
        
    def test_max_vectors(self):
        vectorizer = Vectorizer(score_attr="freq_score")
        m = vectorizer(self.graph, self.vocab)
        # 1: the --> who, the, which, that
        assert m[0].nnz == 0
        # 2: europaische --> European, continental
        assert m[1].nnz == 1
        assert m[1, self.vocab["European"]] == 1.0
        # 3: Union --> union
        assert m[2].nnz == 1
        assert m[2, self.vocab["union"]] == 1.0
        # 4: hat --> own, have, bear, entertain, experience, gotta
        # NB "have" has highest score but is not in vocab,
        # so translations vector is empty...
        assert m[3].nnz == 0
        
    def test_min_vectors(self):
        vectorizer = Vectorizer(score_attr="freq_score", min_score=0.02)
        m = vectorizer(self.graph, self.vocab)
        # 1: the --> who, the, which, that
        # NB none of these translations are in the vocabulary
        assert m[0].nnz == 0
        # 2: europaische --> European:0.95, continental:=.05
        assert m[1].nnz == 2
        assert m[1, self.vocab["European"]] == 1.0
        assert m[1, self.vocab["continental"]] == 1.0
        # 3: Union --> union:1.0
        assert m[2].nnz == 1
        assert m[2, self.vocab["union"]] == 1.0
        # 4: hat --> own:0.06, have:0.89, bear:0.019..., entertain:0.0, 
        #            experience:0.03, gotta:0.0
        # NB "have" and "own" are not in vocab
        assert m[3, self.vocab["experience"]] == 1.0


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
        self._classifier_score()
        
    def test_classifier_score_mft(self):
        vectorizer = Vectorizer(score_attr="freq_score")
        self._classifier_score(vectorizer)
    
    def _classifier_score(self, vectorizer=None):
        # make a scorer that uses this classifier
        class_score = ClassifierScore(self.classifier, 
                                      vectorizer=vectorizer)
        
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
