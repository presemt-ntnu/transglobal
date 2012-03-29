"""
test translation dictionary
"""

from nose.tools import raises

from tg.config import config
from tg.transdict import TransDict



class TestTransdict:
    """
    Note that these test are slow because loading the large pickled
    dictionary takes along time
    """
    
    @classmethod
    def setup_class(cls):
        dict_fname = config["dict"]["en-de"]["pkl_fname"]
        print "loading picked dictionary from " + dict_fname
        cls.trans_dict = TransDict.load(dict_fname)         
        # remove the POS mapping
        cls.trans_dict.pos_map = None
            
    def test_lookup_lempos(self):
        lempos, translations = self.trans_dict.lookup_lempos("linguist/n")
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations    

    def test_lookup_lemma(self):
        results = list(self.trans_dict.lookup_lemma("linguist"))
        assert len(results) == 1
        lempos, translations = results[0]
        assert lempos == 'linguist/n'
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations 

    def test_lookup_lemma_2(self):
        results = list(self.trans_dict.lookup_lemma("walk"))
        assert len(results) == 2
        
        lempos, translations = results[0]
        assert lempos == 'walk/n'
        assert translations == self.trans_dict.lookup_lempos(lempos)[1]
        
        lempos, translations = results[1]
        assert lempos == 'walk/vv'
        assert translations == self.trans_dict.lookup_lempos(lempos)[1]
        
    @raises(KeyError)  
    def test_unkown_lemma(self):
        self.trans_dict.lookup_lemma("1q84")
        
    @raises(KeyError)  
    def test_unknown_pos(self):
        self.trans_dict.lookup_lempos("linguist/xyz")
        
    def test_lookup_lempos_mwu(self):
        # 2nd POS tag is indeed incorrect in dict
        lempos, translations = self.trans_dict.lookup_lempos("kick/vv out/vv")
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    def test_lookup_lemma_mwu(self):
        results = list(self.trans_dict.lookup_lemma("kick out"))
        assert len(results) == 1
        lempos, translations = results[0]
        assert lempos == "kick/vv out/vv"
        assert translations == self.trans_dict.lookup_lempos(lempos)[1]
        
    def test_getitem_lempos(self):
        assert list(self.trans_dict["linguist/n"]) == [self.trans_dict.lookup_lempos("linguist/n")]
        
    def test_getitem_lemma(self):
        assert list(self.trans_dict["linguist"]) == list(self.trans_dict.lookup_lemma("linguist"))
        
    def test_getitem_lempos_unknown_pos(self):
        # should backoff to lemma
        assert list(self.trans_dict["linguist/xyz"]) == list(self.trans_dict.lookup_lemma("linguist"))
        
    def test_getitem_lempos_mwu_unknown_pos(self):
        # should backoff to lemma
        assert list(self.trans_dict["kick/xx out/yy"]) == list(self.trans_dict.lookup_lemma("kick out"))
        
    @raises(KeyError)  
    def test_getitem_unkown_lemma(self):
        self.trans_dict["1q84"]

        
        
    
class TestMappedDict:
    
    @classmethod
    def setup_class(cls):
        dict_fname = config["dict"]["en-de"]["pkl_fname"]
        print "loading picked dictionary from " + dict_fname
        cls.trans_dict = TransDict.load(dict_fname)         
        
    def test_lookup_lempos(self):
        # NN -> n
        lempos, translations = self.trans_dict.lookup_lempos("linguist/NN")
        # lempos must contain mapped pos (i.e. lexicon pos)
        assert lempos == 'linguist/n'
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations    
    
        # NP -> n
        assert ( self.trans_dict.lookup_lempos("linguist/NP") == 
                 self.trans_dict.lookup_lempos("linguist/NN") )

    def test_lookup_lemma(self):
        results = list(self.trans_dict.lookup_lemma("linguist"))
        assert len(results) == 1
        lempos, translations = results[0]
        assert lempos == 'linguist/n'
        assert len(translations) == 2
        assert "Linguist/n" in translations
        assert "Sprachwissenschaftler/n" in translations 
    
    @raises(KeyError)  
    def test_unkown_lemma(self):
        self.trans_dict.lookup_lemma("1q84")
        
    @raises(KeyError)  
    def test_unknown_pos(self):
        self.trans_dict.lookup_lempos("linguist/xyz")  
        
    def test_pos_relevancy(self):
        # NN -> n
        lempos, translations = self.trans_dict.lookup_lempos("rerun/NN")
        assert lempos == "rerun/n"
        assert len(translations) == 3
        
        # VB -> v
        lempos, translations = self.trans_dict.lookup_lempos("rerun/VB")
        assert lempos == "rerun/vv"
        assert len(translations) == 5
              
        # nouns and verbs combined
        all_translations = []
        for lempos, translations in self.trans_dict.lookup_lemma("rerun"):
            all_translations += translations
        assert len(all_translations) == 8
        
    def test_lookup_lempos_mwu(self):
        # VB -> vv
        # 2nd POS tag (vv) is indeed incorrect in dict
        lempos, translations = self.trans_dict.lookup_lempos("kick/VB out/VB")
        # lempos must contain mapped pos (i.e. lexicon pos)
        assert lempos == "kick/vv out/vv"
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    def test_lookup_lemma_mwu(self):
        lempos, translations = list(self.trans_dict.lookup_lemma("kick out"))[0]
        assert len(translations) == 2
        assert "hinauswerfen/v*.full" in translations
        assert "werfen/v*.full hinaus/v*.full" in translations
        
    @raises(KeyError)
    def test_unkown_lemma_mwu(self):
        self.trans_dict.lookup_lemma("1q84 out")
        
        
                    
if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)    