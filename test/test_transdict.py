"""
test translation dictionary
"""

from nose.tools import raises

from tg.config import config
from tg.transdict import TransDict


class Test_Transdict:
    """
    Note that these test are slow because loading the large pickled
    dictionary takes along time
    """
    
    @classmethod
    def setup_class(cls):
        dict_fname = config["en-de_dict_pkl"]
        print "loading picked dictionary from " + dict_fname
        cls.trans_dict = TransDict.load(dict_fname) 

    def test_single_lemma(self):
        translations = self.trans_dict["linguist"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
        
    def test_single_lempos(self):
        translations = self.trans_dict["linguist/n"]
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations
      
    @raises(KeyError)  
    def test_oov_lemma(self):
        self.trans_dict["1q84"]
        
    def test_mwu_lemma(self):
        translations = self.trans_dict["kick out"]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    def test_mwu_lemma(self):
        # 2nd POS tag is indeed incorrect in dict
        translations = self.trans_dict["kick/vv out/vv"]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    
