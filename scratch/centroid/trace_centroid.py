"""
Ad-hoc code for tracing nearest centroid classification

This is the result of "exploratory coding" and therefore messy stuff. 
Is there a way to integrate this properly in the NearestCentroid classes?
"""

import codecs
import cPickle
import logging
import sys

import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import normalize

from tg.store import DisambiguatorStore
from tg.utils import set_default_log
from tg.classcore import DE_CONTENT_POS

log = logging.getLogger(__name__)


def trace_nc(graphs_fname, model_fname, n=None, source_pos=[],
             outf=codecs.getwriter('utf8')(sys.stdout)):
    if isinstance(outf, basestring):
        outf = codecs.open(outf, "w", encoding="utf-8")
    graphs = cPickle.load(open(graphs_fname))
    model = DisambiguatorStore(model_fname)
    estimator = model.load_estimator()
    vocab = np.array(model.load_vocab())
    vocab_dict = model.load_vocab(as_dict=True)
    score_attr = "centroid_score"
    
    for graph_count, graph in enumerate(graphs[:n]):
        source_string = graph.source_string()
        outf.write( 100 * "=" + "/n")
        outf.write( u"{}: {}\n".format(graph_count + 1, source_string))
        outf.write( 100 * "=" + "\n\n")
        
        reverse_lookup = make_reverse_lookup(graph)
        source_node_vectors = make_source_node_vectors(graph, vocab_dict)
        source_graph_vector = sp.csr_matrix(source_node_vectors.sum(axis=0))
        
        for sn, node_vec in zip(graph.source_nodes_iter(ordered=True), 
                               source_node_vectors):
            if source_pos and graph.pos(sn) not in source_pos:
                continue
            
            try:
                source_lempos = u" ".join(graph.node[sn]["lex_lempos"])
            except KeyError:
                log.debug(u"(annotated) source lempos {0} not in"
                          u"lexicon\n".format(graph.lempos(sn)))
                continue
            
            try:
                model.restore_fit(source_lempos, estimator)
            except KeyError:
                log.debug(u"no model available for (lexicon) source lempos "
                          u"{0}\n".format(source_lempos))
                continue
            
            context_vec = source_graph_vector - node_vec
            context_vec = context_vec.toarray()
            normalize(context_vec, copy=False)
            
            try:
                mask = model.load_vocab_mask(source_lempos)[:]
            except KeyError:
                local_vocab = vocab
            else:
                context_vec = context_vec[:,mask]
                local_vocab = vocab[mask]
            
            try:
                centroids = estimator.centroids_
            except AttributeError:
                # pipeline
                centroids = estimator.steps[-1][-1].centroids_  
                
            target_lempos_list = model.load_target_names(source_lempos)
                           
            outf.write( 100 * "-" + "\n")
            outf.write( source_lempos + "\n")
            outf.write( 100 * "-" + "\n\n")
            
            for target_lempos, target_centroid in zip(target_lempos_list,
                                                      centroids):
                prod = target_centroid * context_vec
                outf.write( u"==> {:<24} {:1.4f}  {}\n".format(
                    target_lempos, 
                    prod.sum(),
                    prod.sum() * 100 * "X"))
            outf.write("\n")
            
            for target_lempos, target_centroid in zip(target_lempos_list,
                                                      centroids):
                prod = target_centroid * context_vec
                indices = target_centroid.argsort()[::-1]
                
                outf.write(  "\n" + source_string + "\n\n")
                
                outf.write( u"{:<64} ==> {:<24} {:1.4f}  {}\n".format(
                    source_lempos,
                    target_lempos, 
                    prod.sum(),
                    prod.sum() * 100 * "X"))
                
                for i in indices:
                    if prod[0,i] > 0:
                        context_lemma = local_vocab[i]
                        sources = ",".join(reverse_lookup[context_lemma])
                        bar = prod[0,i] * 100 * "*"
                        outf.write( u"{:<64} --> {:<24} {:1.4f}  {}\n".format(
                            sources, 
                            context_lemma, 
                            prod[0,i],
                            bar))
                outf.write("\n")
                        

    
def make_source_node_vectors(graph, vocab):
    """ 
    Create a sparse matrix consisting of a vector for every source node.
    The vector for a source node counts its translation candidates
    (target lemmas) that are within the vocabulary.
    """
    # no of source nodes is not known in advance, so allocate too many rows
    dim = (len(graph), len(vocab))
    # lil sparse format allows indexing 
    mat = sp.lil_matrix(dim, dtype="f8")
    
    for row_i, u in enumerate(graph.source_nodes_iter(ordered=True)):
        for u, v, data in graph.trans_edges_iter(u):
            # TODO: handle source/target hypernodes 
            if graph.is_target_node(v):
                target_lemma = graph.lemma(v)
                try:
                    col_j = vocab[target_lemma]
                except KeyError:
                    # ignore target lemma that is out of vocabulary
                    continue
                mat[row_i, col_j] += 1
                
    mat = mat.tocsr()
    # remove superfluous rows now that number of source nodes is known
    return mat[:row_i, :]


def make_reverse_lookup(graph):
    reverse_lookup = {}
    
    for sn, tn, data in graph.trans_edges_iter():
        # FIXME: handle hypernodes
        if graph.is_target_node(tn) and graph.is_source_node(sn):
            source_lempos = graph.lempos(sn)
            target_lemma = graph.lemma(tn)
            reverse_lookup.setdefault(target_lemma, []).append(source_lempos)
                                                           
    return reverse_lookup
        
        


set_default_log(level=logging.INFO)

trace_nc("_centroid_metis_de-en/centroid_metis_de-en_graphs.pkl",
         "_centroid_metis_de-en/centroid_metis_de-en.hdf5",
         #n=10,
         source_pos=DE_CONTENT_POS,
         outf="_centroid_metis_de-en/centroid_metis_de-en_trace.txt"
         )
    
    
    