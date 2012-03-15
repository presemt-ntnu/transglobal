"""
test translation dictionary
"""

from nose.tools import raises

from tg.config import config
from tg.transdict import TransDict, DictAdaptor


trans_dict = None

def setup_module():
    global trans_dict
    dict_fname = config["dict"]["en-de"]["pkl_fname"]
    print "loading picked dictionary from " + dict_fname
    trans_dict = TransDict.load(dict_fname) 
    
    
def teardown_module():
    global trans_dict
    trans_dict = None
    


class TestTransdict:
    """
    Note that these test are slow because loading the large pickled
    dictionary takes along time
    """
    
    def test_single_lemma(self):
        translations = trans_dict["linguist"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
        
    def test_single_lempos(self):
        translations = trans_dict["linguist/n"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
      
    @raises(KeyError)  
    def test_unkown_lemma(self):
        trans_dict["1q84"]
        
    @raises(KeyError)  
    def test_unknown_pos(self):
        trans_dict["linguist/xyz"]
        
    def test_mwu_lemma(self):
        translations = trans_dict["kick out"]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    def test_mwu_lempos(self):
        # 2nd POS tag is indeed incorrect in dict
        translations = trans_dict["kick/vv out/vv"]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
        
    
class TestDictAdaptor:
    
    @classmethod
    def setup_class(cls):
        cls.mapped_dict = DictAdaptor(trans_dict,
                                      config["dict"]["en-de"]["posmap_fname"])
    
    def test_single_lemma(self):
        translations = self.mapped_dict["linguist"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
        
    def test_single_lempos(self):
        # NN -> n
        translations = self.mapped_dict["linguist/NN"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
        
        # NP -> n
        translations = self.mapped_dict["linguist/NP"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
                
    @raises(KeyError)  
    def test_unkown_lemma(self):
        self.mapped_dict["1q84"]
        
    @raises(KeyError)
    def test_unknown_lemma_pos(self):
        self.mapped_dict["1q84/xyz"]
        
    def test_unknown_pos(self):        
        # backs off to lemma, ignoring pos
        translations = self.mapped_dict["linguist/xyz"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
        
    def test_pos_relevancy(self):
        # NN -> n
        translations = self.mapped_dict["rerun/NN"]
        assert len(translations) == 3
        
        # VB -> v
        translations = self.mapped_dict["rerun/VB"]
        assert len(translations) == 5
              
        # nouns and verbs combined
        translations = self.mapped_dict["rerun"]
        assert len(translations) == 8
        
    def test_mwu_lemma(self):
        translations = self.mapped_dict["kick out"]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    def test_mwu_lempos(self):
        # VB -> vv
        # 2nd POS tag (vv) is indeed incorrect in dict
        translations = self.mapped_dict["kick/VB out/VB"]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    @raises(KeyError)
    def test_mwu_unkown_lemma(self):
        translations = self.mapped_dict["1q84 out"]
        
    def test_mwu_unkown_pos(self):    
        # backs off to lemma, ignoring pos, yielding same result (in this
        # case)
        translations = self.mapped_dict["kick/xyz out/xyz"]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    