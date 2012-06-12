"""
test annotation
"""

from codecs import open

from tg.annot import ( TreeTaggerEnglish, TreeTaggerGerman, ILSP_NLP_Greek,
                       OsloBergenTagger )

from setup import test_data_dir



def check_graph_format(graph_list): 
    # all sentences?
    assert len(graph_list) ==  7 
    
    # each node has word, lemma and tag attributes?
    for graph in graph_list:
        for node_data in graph.node.values():
            assert node_data["word"]
            assert node_data["lemma"]
            assert node_data["pos"]
    
    
class TestTreetaggerEnglish:
    
    def test_annot_text_file(self):
        text_fname = test_data_dir + "/sample_en_1.txt"
        annotator = TreeTaggerEnglish()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = test_data_dir + "/sample_en_1.xml"
        annotator = TreeTaggerEnglish()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)
       
class TestTreetaggerGerman:
    
    def test_annot_text_file(self):
        text_fname = test_data_dir + "/sample_de_1.txt"
        annotator = TreeTaggerGerman()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = test_data_dir + "/sample_de_1.xml"
        annotator = TreeTaggerGerman()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)
            
            
            
class TestILSP_NLP_Greek:           
    """
    This test relies on a webservice and won't work without an network
    connection
    """
    
    def test_annot_text_file(self):
        text_fname = test_data_dir + "/sample_gr_1.txt"
        annotator = ILSP_NLP_Greek()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = test_data_dir + "/sample_gr_1.xml"
        annotator = ILSP_NLP_Greek()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)
    

class TestOsloBergenTagger:
    
    def test_annot_text_file(self):
        text_fname = test_data_dir + "/sample_no_1.txt"
        annotator = OsloBergenTagger()
        graph_list = annotator.annot_text_file(text_fname)
        check_graph_format(graph_list)
            
    def test_annot_xml_file(self):
        xml_fname = test_data_dir + "/sample_no_1.xml"
        annotator = OsloBergenTagger()
        graph_list = annotator.annot_xml_file(xml_fname)
        check_graph_format(graph_list)    
    
            
            
if __name__ == "__main__":
    import nose, sys
    sys.argv.append("-v")
    nose.run(defaultTest=__name__)
    ##import logging
    ##from tg.utils import set_default_log
    ##set_default_log()
    ##logging.getLogger("tg.annot").setLevel(logging.DEBUG)
    ##TestOsloBergenTagger().test_annot_text_file()
    ##TestOsloBergenTagger().test_annot_xml_file()

    
    