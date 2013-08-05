# -*- coding: utf-8 -*-


from tg.ambig import AmbiguityMap
from tg.config import config

class TestAmbguityMap:
    
    ambig_fname = config["test_data_dir"] +"/de-en_ambig.tab"    
    
    def test_init_from_file(self):
        am = AmbiguityMap(self.ambig_fname)
        assert len(list(am.source_target_pair_iter())) == 11
        assert am["köstlich/adj".decode("utf-8")] == ["delicious/jj".decode("utf-8"),
                                                       "rich/jj".decode("utf-8")]
        
    def test_subset(self):
        subset = {"köstlich/adj".decode("utf-8")}
        am = AmbiguityMap(self.ambig_fname, subset=subset)
        assert len(list(am.source_target_pair_iter())) == 2
        
    def test_graphs(self):
        # "absolute/adj" is the only lempos shared between graphs and ambiguity table
        graphs_fname = config["test_data_dir"] +"/graphs_sample_out_de-en.pkl" 
        subset = AmbiguityMap.extract_source_lempos_subset(graphs_fname)
        am = AmbiguityMap(self.ambig_fname, subset=subset)
        assert len(am) == 1
        assert am["absolut/adj".decode("utf-8")] == ["absolute/jj".decode("utf-8"),
                                                     "thorough/jj".decode("utf-8"),
                                                     "total/jj".decode("utf-8")]
        
        
        
        
    
        

        
        
        
        

