"""
report translation differences
"""

import codecs
import cPickle
import sys
from collections import OrderedDict

from tg.mteval import read_ref_trans, read_ref_trans_counts


def trans_diff(inf, score_attrs, ref_fname=None, colwidth=32,
               outf=codecs.getwriter('utf8')(sys.stdout)):
    """
    Report translation differences
    
    Outputs all cases where translations differ when selected on score_attr.
    If reference translations are provided, it also shows the reference
    translation sentences as well as a guess of the reference lemma(s) per
    source lemma.
    
    Parameters
    ----------
    inf: list or str
        list of TransGraph instances or filename of pickled graphs
    score_attrs: list of strings
        list of scoring attributes
    ref_fname: str
        filename of reference translations in mteval xml format
    col_width: int
        column width
    outf: file or str
        file or filename for output
        
    Notes
    -----
    Does not support multi-word expressions
    """
    assert len(score_attrs) > 1
    
    if isinstance(inf, basestring):
        inf = cPickle.load(open(inf))
        
    if isinstance(outf, basestring):
        outf = codecs.open(outf, "w", encoding="utf-8")
        
    no_cols = 1 + len(score_attrs)
    
    if ref_fname:
        ref_trans = read_ref_trans(ref_fname, flatten=True)
        ref_counts = read_ref_trans_counts(ref_fname, flatten=True)
        no_cols += 1
    else:
        ref_lemmas = set()
    
    bar = no_cols * colwidth * u"=" + u"\n"
    subbar = no_cols * colwidth * u"-" + u"\n"        
    
    for i, graph in enumerate(inf):
        diffs = graph_trans_diff(graph, score_attrs)
        if not diffs:
            continue
        
        outf.write(bar)
        outf.write( u"SEGMENT {} (id={})\n".format(graph.graph.get("n"),
                                                   graph.graph.get("id")))
        outf.write(bar + u"\n")
        outf.write(u"SRC:   {}\n".format(
            graph.source_string()))
        if ref_fname: 
            for ref_lemmas in ref_trans[i]:
                outf.write(u"REF:   {}\n".format(ref_lemmas))
                
        outf.write(u"\n")
        outf.write(u"SRC LEMPOS:".ljust(colwidth))
        for attr in score_attrs:
            outf.write((attr.upper() + ":").ljust(colwidth))
        if ref_fname:
            outf.write(u"REF TRANS:".ljust(colwidth))
        outf.write(u"\n" + subbar)         
        
        for source_node, max_scores in diffs.iteritems():
            if ref_fname:
                ref_lemmas = get_ref_lemmas(graph, source_node, ref_counts[i])
                        
            outf.write(graph.lempos(source_node).ljust(colwidth))
                       
            for score, target_node in max_scores:
                if score is not None:
                    target_lemma = graph.lemma(target_node)
                else:
                    target_lemma = u"__NONE__"
                    
                pair = u"{}: {:.4f}: {}".format(
                    "+" if target_lemma in ref_lemmas else "-",
                    score,
                    target_lemma)
                outf.write(pair.ljust(colwidth))
                
            if ref_fname:
                outf.write(", ".join(ref_lemmas or ["---"]))
                
            outf.write(u"\n")
        outf.write(u"\n")


def get_ref_lemmas(graph, source_node, ref_counts):
    """
    Get target lemmas which also occur in reference translation(s)
    """
    ref_lemmas = set() 
    for _, target_node, _ in graph.trans_edges_iter(source_node): 
        target_lemma = graph.lemma(target_node).lower()
        if target_lemma in ref_counts:
            ref_lemmas.add(target_lemma)   
    return ref_lemmas


def graph_trans_diff(graph, score_attrs):
    """
    Figure out for which source nodes in graph the max scores on score_attrs
    result in different translations
    """
    diffs = OrderedDict()
    # TODO: handle hypernodes
    for source_node in graph.source_nodes_iter(ordered=True):
        unique_nodes = set()
        max_scores = []
                    
        for attr in score_attrs:
            score, target_node = graph.max_score(source_node, attr)
            max_scores.append((score, target_node))
            if score is not None:
                unique_nodes.add(target_node)
                
        if len(unique_nodes) > 1:
            diffs[source_node] = max_scores
            
    return diffs  
        
    
    
    