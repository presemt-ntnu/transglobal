"""
test annotation
"""

from codecs import open

from tg.annot import TreeTaggerEnglish, TreeTaggerGerman, ILSP_NLP_Greek

from setup import test_data_dir


def test_treetagger_english():
    text_fname = test_data_dir + "/sample_en_1.txt"
    text = open(text_fname, encoding="utf-8").read()
    annotator = TreeTaggerEnglish()
    graph_list = annotator(text)
    
    # all sentences?
    assert len(graph_list) ==  7 
    
    # each node has word, lemma and tag attributes?
    for graph in graph_list:
        for node_data in graph.node.values():
            assert node_data["word"]
            assert node_data["lemma"]
            assert node_data["pos"]
            
            
def test_treetagger_german():
    text_fname = test_data_dir + "/sample_de_1.txt"
    text = open(text_fname, encoding="utf-8").read()
    annotator = TreeTaggerGerman()
    graph_list = annotator(text)
    
    # all sentences?
    assert len(graph_list) ==  7 
    
    # each node has word, lemma and tag attributes?
    for graph in graph_list:
        for node_data in graph.node.values():
            assert node_data["word"]
            assert node_data["lemma"]
            assert node_data["pos"]
            
            
def test_ilsp_nlp_greek():
    """
    This test relies on a webservice and won't work without an network
    connection
    """
    text_fname = test_data_dir + "/sample_gr_1.txt"
    text = open(text_fname, encoding="utf-8").read()
    annotator = ILSP_NLP_Greek()
    graph_list = annotator(text)
    
    # all sentences?
    assert len(graph_list) ==  7 
    
    # each node has word, lemma and tag attributes?
    for graph in graph_list:
        for node_data in graph.node.values():
            assert node_data["word"]
            assert node_data["lemma"]
            assert node_data["pos"]
    
            
            
if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
    
    
    