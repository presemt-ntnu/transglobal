# -*- coding: utf-8 -*-

"""
test annotation
"""

from codecs import open

from tg.config import config
from tg.annot import ( TreeTaggerEnglish, 
                       TreeTaggerGerman, 
                       ILSP_NLP_Greek,
                       OsloBergenTagger )

# HACK: suds logging triggers an exception in suds.max.core
import logging
suds_log = logging.getLogger("suds")
suds_log.setLevel(logging.ERROR)


def check_graph_format(graph_list): 
    # all sentences?
    assert len(graph_list) ==  7 
    
    # each node has word, lemma and tag attributes?
    for graph in graph_list:
        assert isinstance(graph.graph["n"], int)
        assert graph.graph["id"] is None or isinstance(graph.graph["id"], str)
        for node_data in graph.node.values():
            assert node_data["word"]
            assert node_data["lemma"]
            assert node_data["pos"]
    
    
class TestTreetaggerEnglish:
    
    def test_annot_text_file(self):
        text_fname = config["test_data_dir"] + "/sample_en_1.txt"
        annotator = TreeTaggerEnglish()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = config["test_data_dir"] + "/sample_en_1.xml"
        annotator = TreeTaggerEnglish()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)
        
    def test_annot_sentences(self):
        sentences = [
            "It may seem obvious to just create one language for  "
            "everybody to use.",
            "Luckily, several linguists felt the same way.",
            "They made up what we call constructed languages.",
            "But, languages re a big part of a people's culture "
            "and identity and most of them have long interesting histories.",
            "People aren't willing to give them up.",
            "It is also very hard to become fluent in a language.",
            "It may seem natural to you to speak English, but it is actually " 
            "very hard for many adults to learn." ]
        ids = "a", "b", "c", "d", "e", "f", "g"
        annotator = TreeTaggerEnglish()
        graph_list = annotator.annot_sentences(sentences, encoding="utf-8",
                                               ids=ids)
        check_graph_format(graph_list)
        # check graph id's
        for graph, id in zip(graph_list, ids):
            assert graph.graph["id"] == id
        
        
       
       
class TestTreetaggerGerman:
    
    def test_annot_text_file(self):
        text_fname = config["test_data_dir"] + "/sample_de_1.txt"
        annotator = TreeTaggerGerman()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = config["test_data_dir"] + "/sample_de_1.xml"
        annotator = TreeTaggerGerman()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)
            
                       
class TestILSP_NLP_Greek:
    """
    This test relies on a webservice and won't work without an network
    connection
    """
    
    def test_annot_text_file(self):
        text_fname = config["test_data_dir"] + "/sample_gr_1.txt"
        annotator = ILSP_NLP_Greek()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = config["test_data_dir"] + "/sample_gr_1.xml"
        annotator = ILSP_NLP_Greek()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)
    

class TestOsloBergenTagger:
    
    def test_annot_text_file(self):
        text_fname = config["test_data_dir"] + "/sample_no_1.txt"
        annotator = OsloBergenTagger()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = config["test_data_dir"] + "/sample_no_1.xml"
        annotator = OsloBergenTagger()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)    
        
    def test_annot_sentences(self):
        sentences = [
            "Den utskremte ungdommen hos forhandleren ville ikke ta tilbake "
            "skoene fordi de var så skitne.",
            "For at denne politikken skulle lykkes, var det imidlertid "
            "nødvendig å stanse kapprustningen og få til reelle "
            "nedrustningsavtaler.",
            "For mer informasjon, besøk greendayacrosstheworld.org .",
            "Billie Joe twitret senere om videoen også funnet på GDA .",
            "Dette er av våre kloke urbefolkninger blitt kalt Endetiden.",
            "Et problem med såpe er at bare alkalisaltene av de store "
            "fettsyrene er vannløselige.",
            "Flertallets innstilling fikk ved behandlingen i Stortinget støtte "
            "av Venstre." ]
        ids = "a", "b", "c", "d", "e", "f", "g"
        annotator = OsloBergenTagger()
        graph_list = annotator.annot_sentences(sentences, encoding="utf-8",
                                               ids=ids)
        check_graph_format(graph_list)
        # check graph id's
        for graph, id in zip(graph_list, ids):
            assert graph.graph["id"] == id
    
    
    