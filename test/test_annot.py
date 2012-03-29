"""
test annotation
"""

from codecs import open

from tg.annot import TreeTaggerEnglish

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
            
            
if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
    
    
    