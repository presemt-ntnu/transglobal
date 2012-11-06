# -*- coding: utf-8 -*-

"""
test lookup
"""

from cPickle import load

from tg.config import config
from tg.annot import TreeTaggerGerman
from tg.lookup import Lookup


class TestLookup:

    @classmethod
    def setup_class(cls):
        # make annotated graphs
        src_fname = config["test_data_dir"] + "/sample_out_de-en.src"
        annotator = TreeTaggerGerman()
        cls.graphs = annotator.annot_xml_file(src_fname) 
        # load minimal mapped translation dictionary
        en_de_dict_pkl_fname = ( config["test_data_dir"] + 
                                 "/dict_sample_out_de-en.pkl" )
        cls.dict = load(open(en_de_dict_pkl_fname))
        # perform lookup
        lookup = Lookup(cls.dict)
        lookup(cls.graphs)
        
        
    def test_lookup(self):
        for graph in self.graphs:
            for sn in graph.source_nodes_iter(ordered=True):
                assert ( self._translations_in_graph(graph, sn) ==
                         self._translations_in_dict(graph, sn) )

    def _translations_in_graph(self, graph, sn):
        """
        get all translation for source node from dictionary
        """
        return set(graph.lempos(tn)
                for sn, tn, _ in graph.trans_edges_iter(sn) )
    
    def _translations_in_dict(self, graph, sn):
        """
        get all translation for source node from graph
        """
        return set(t
                   for p in self.dict.get(graph.lempos(sn), []) for t in p[1] ) 