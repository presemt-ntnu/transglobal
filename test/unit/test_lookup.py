# -*- coding: utf-8 -*-

"""
test lookup
"""

from cPickle import load

from tg.lookup import Lookup

from setup import annot_graphs_en_pkl_fname, en_de_dict_pkl_fname


class TestLookup:

    def setUp(self):
        # load annotated graphs
        self.graph_list = load(open(annot_graphs_en_pkl_fname))
        # load minimal mapped translation dictionary
        self.en_de_dict = load(open(en_de_dict_pkl_fname))
        # perform lookup
        lookup = Lookup(self.en_de_dict)
        lookup(self.graph_list)
        
    def test_lookup_1(self):
        # check translations of "linguist"
        graph = self.graph_list[1]
        sn = "s4"
        assert graph.node[sn]["lemma"] == "linguist"
        graph.trans_edges_iter(sn)
        translations = [ graph.node[tn]["lemma"]
                         for _, tn, _ in  graph.trans_edges_iter(sn) ]
        translations.sort()
        assert translations == ["Linguist", "Sprachwissenschaftler"]
        
    def test_lookup_2(self):
        # check translations of "luckily"
        graph = self.graph_list[1]
        sn = "s1"
        assert graph.node[sn]["lemma"] == "luckily"
        graph.trans_edges_iter(sn)
        tn1, tn2 = [ tn for _, tn, _ in  graph.trans_edges_iter(sn) ]
        # first target node is a hypernode realizing a multi-word translation
        assert graph.lemma(tn1) == "zum Glück".decode("utf8")
        assert graph.lemma(tn2) == "glücklicherweise".decode("utf-8")
        
        
        
                            
if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)            
        
            
        
    