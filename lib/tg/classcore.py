# -*- coding: utf-8 -*-

"""
scoring translation candidates with a classifier
"""

import logging

import numpy as np
import operator
import scipy.sparse as sp

from tg.graphproc import GraphProcess

log = logging.getLogger(__name__)

# uncomment line to below to enable logging
# log.setLevel(logging.DEBUG)


class ClassifierScore(GraphProcess):
    """
    Add classifier scores to translation candidates.
    
    Requires a trained classifier that, given a lempos and context vector,
    assigns scores to translation candidates. Takes a list of graphs (or a
    single graph) and applies the classifier to score translation candidates.
    Scores are added to the attribute called `score_attr` on translation
    edges. Scores are normalized so they sum to one.
        
    Parameters
    ----------
    classifier: `TranslationClassifier` instance
        Trained classifier which takes a context vector and assigns a score to
        each translation candidate
    score_attr: str, optional
        Attribute on translation edges which holds the classifier score
    filter: function, optional
        A function to filter out certain source nodes. It must take two 
        arguments: a graph and a source node. If its return value is true,
        the source node will not be scored by the classifier.
    vectorizer: {"full", "mft"}
        Method of constructing context vectors
        * "full" (default): use all possible translations of source words
        * "mft": use only most frequent translation of each source word; 
                 requires prior scoring with FreqScore
        
    Notes
    -----
    Use generic __call__ method on parent `GraphProc` to add scores to a 
    single graph or a list of graphs.
    """
    
    def __init__(self, classifier, score_attr="class_score", filter=None,
                 vectorizer = "full"):
        self.classifier = classifier
        self.score_attr = score_attr
        if filter:
            self.filter = filter
        else:
            self.filter = lambda graph, node : False
            
        if vectorizer == "full":
            self._make_source_node_vectors = self._make_full_vectors
        elif vectorizer == "mft":
            self._make_source_node_vectors = self._make_mft_vectors
        else:
            raise ValueError("unknown value '{}' for keyword argument "
            "'vectorizer'".format(vectorizer))
        
        if log.isEnabledFor(logging.debug): self._construct_reverse_vocab()
    
    def _single_run(self, graph):
        """
        Add classifier scores to translation edges in graph
        """
        log.info("applying {0} to graph {1}".format(
            self.__class__.__name__,
            graph.graph["id"]))
        
        source_node_vectors = self._make_source_node_vectors(graph)
        # The context vector for the whole source graph is obtained by
        # summing the vectors for the source nodes. As sum() returns a dense
        # vector, it needs to be converted to sparse format.
        source_graph_vector = sp.csr_matrix(source_node_vectors.sum(axis=0))
        
        if log.isEnabledFor(logging.debug): 
            self._log_graph_vector(graph, source_graph_vector)
        
        for u, node_vec in zip(graph.source_nodes_iter(ordered=True), 
                               source_node_vectors):
            try:
                # classifier selects classification model on the basis of the
                # *lexicon's* lempos (instead of the tagger's POS)
                source_lempos = u" ".join(graph.node[u]["lex_lempos"])
            except KeyError:
                log.debug(u"(annotated) source lempos {0} not in"
                          u"lexicon\n".format(graph.lempos(u)))
                continue
            
            # In order to obtain the context vector required for
            # classification, take the source graph vector and the current
            # source node vector. In efect, this removes the counts of the
            # translation candidates for the current source node from source
            # graph vector.
            context_vec = source_graph_vector - node_vec
            # Get scores from classifier for each target lempos
            lempos2score = self.classifier.score(source_lempos, context_vec)
            
            if not lempos2score:
                log.debug(u"no model available for (lexicon) source lempos "
                          u"{0}\n".format(source_lempos))
                continue
            
            if self.filter(graph, u):
                log.debug(u"filtering out source lempos {0}\n".format(
                    graph.lempos(u)))
                continue
            
            if log.isEnabledFor(logging.debug):
                self._log_source_node_vector(node_vec, source_lempos)
                self._log_context_vector(context_vec, source_lempos)
                self._log_scores(lempos2score, source_lempos)
                
            self._add_scores(graph, u, lempos2score)

    def _add_scores(self, graph, u, lempos2score):
        """
        Add scores for translation candidates of source node `u` to its
        translation edges. Also store target node of the best translation to
        the best_nodes dict on the source node.
        """
        best_score, best_node = None, None
        
        for u, v, data in graph.trans_edges_iter(u):
            # TODO: handle source/target hypernodes 
            if graph.is_target_node(v):
                target_lempos = graph.lempos(v)
                
                try:
                    score = data[self.score_attr] = lempos2score[target_lempos]
                except KeyError:
                    # model does not predict this target lemma,
                    # (which may be different from a 0.0 score)
                    continue
                
                if score > best_score:
                    best_score, best_node = score, v

        if "best_nodes" not in graph.node[u]:
            graph.node[u]["best_nodes"] = {}
        
        graph.node[u]["best_nodes"][self.score_attr] = best_node
           
    def _make_full_vectors(self, graph):
        """ 
        Create a sparse matrix consisting of a vector for every source node.
        The vector for a source node counts its translation candidates
        (target lemmas) that are within the vocabulary.
        """
        # no of source nodes is not known in advance, so allocate too many rows
        dim = (len(graph), len(self.classifier.vocab))
        # lil sparse format allows indexing 
        mat = sp.lil_matrix(dim, dtype="f8")
        
        for row_i, u in enumerate(graph.source_nodes_iter(ordered=True)):
            for u, v, data in graph.trans_edges_iter(u):
                # TODO: handle source/target hypernodes 
                if graph.is_target_node(v):
                    target_lemma = graph.lemma(v)
                    try:
                        col_j = self.classifier.vocab[target_lemma]
                    except KeyError:
                        # ignore target lemma that is out of vocabulary
                        continue
                    mat[row_i, col_j] += 1
                    
        mat = mat.tocsr()
        # remove superfluous rows now that number of source nodes is known
        return mat[:row_i + 1, :]
    
    def _make_mft_vectors(self, graph):
        """ 
        Create a sparse matrix consisting of a vector for every source node.
        The vector for a source node indicates its most frequent translation candidate.
        Application of this method assumes that FreqScore has been applied and that a
        'freq_score' attribute is present on edges. 
        """
        # no of source nodes is not known in advance, so allocate too many rows
        dim = (len(graph), len(self.classifier.vocab))
        # lil sparse format allows indexing 
        mat = sp.lil_matrix(dim, dtype="f8")
        freq_score = "freq_score"
        
        for row_i, u in enumerate(graph.source_nodes_iter(ordered=True)):
            # TODO: handle source/target hypernodes             
            score, v = graph.max_score(u, freq_score)
            
            # if v is None, then there are no translation with freq_score
            # attribute or no translation edges at all
            if v:
                target_lemma = graph.lemma(v)
                
                try:
                    col_j = self.classifier.vocab[target_lemma]
                except KeyError:
                    # ignore target lemma that is out of vocabulary
                    continue
                
                mat[row_i, col_j] += 1
                    
        mat = mat.tocsr()
        # remove superfluous rows now that number of source nodes is known
        return mat[:row_i + 1, :]
    
    # tracing/debugging code
    
    def _construct_reverse_vocab(self):
        self.reversed_vocab = zip(*sorted(self.classifier.vocab.items(), 
                                          key=operator.itemgetter(1)))[0]

    def _log_graph_vector(self, graph, source_graph_vector):
        log.debug("source sentence lemmas: " + " ".join(graph.source_lemmas()))
        items = [ u"{0}={1}".format(self.reversed_vocab[i], count)
                  for i, count in zip(source_graph_vector.indices, 
                                      source_graph_vector.data) ]
        items_str = u", ".join(items)
        log.debug(u"source graph vector: {0}\n".format(items_str))

    def _log_source_node_vector(self, node_vec, source_lempos):
        items = [ u"{0}={1}".format(self.reversed_vocab[i], count)
                  for i, count in zip(node_vec.indices, node_vec.data) ]
        items_str = u", ".join(items)
        log.debug(u"source node vector for lempos {0}: {1}".format(
            source_lempos, items_str))
        
    def _log_context_vector(self, context_vec, source_lempos):
        items = [ u"{0}={1}".format(self.reversed_vocab[i], count)
                  for i, count in zip(context_vec.indices, context_vec.data) ]
        items_str = u", ".join(items)
        log.debug(u"context vector for lempos {0}: {1}".format(
            source_lempos, items_str))
        
    def _log_scores(self, lempos2score, source_lempos):
        items = sorted(lempos2score.items(), reverse=True,
                       key=operator.itemgetter(1))
        items_str = u", ".join(u"{0}={1:.4f}".format(*pair) for pair in items)
        log.debug(u"scores for lempos {0}: {1}\n".format(source_lempos,
                                                         items_str))
                
        
   
    
# ---------------------------------------------------------------------------- 
# Filter functions
# ---------------------------------------------------------------------------- 


def filter_functions(lang):
    if lang == "de":
        return filter_german
    elif lang == "en":
        return filter_en_function_words
    elif lang == "no":
        return filter_no
    elif lang == "gr":
        return filter_gr_function_words
    else:
        raise ValueError("no filter function for language {}".format(lang))



# German

DE_CONTENT_POS = set("ADJA ADJD "
                     "ADV " 
                     #"CARD ITJ "
                     "NN NE "
                     "VVFIN VVIMP VVINF VVIZU VVPP VAFIN VAIMP VAINF VAPP "
                     #"VMFIN VMINF VMPP "
                     .split())

DE_AUX_LEMMA = set(("sein", "haben", "werden"))

def filter_german(graph, node):
    return ( filter_de_aux_verbs(graph, node) or
             filter_de_function_words(graph, node) )

def filter_de_aux_verbs(graph, node):
    """
    filter out Germman auxiliary verbs on the basis of their lemma
    """
    return graph.lemma(node) in DE_AUX_LEMMA

def filter_de_function_words(graph, node):
    """
    filter out Germman function words on the basis of the tagger's POS tag
    """
    return graph.pos(node) not in DE_CONTENT_POS
    
    
    
# English

EN_CONTENT_POS = set("JJ JJR JJS NN NNS NP NPS RB RBR RBS "
                     "VB VBD VBG VBN VBP VBZ".split())

def filter_en_function_words(graph, node):
    """
    filter out English function words on the basis of the tagger's POS tag
    """
    return graph.pos(node) not in EN_CONTENT_POS



# Norwegian

NO_CONTENT_POS = set("subst verb adv adj".split())

NO_AUX_LEMMA = set("være bli få ha".decode("utf-8").split())

def filter_no(graph, node):
    return ( filter_no_aux_verbs(graph, node) or
             filter_no_function_words(graph, node) )

def filter_no_aux_verbs(graph, node):
    """
    filter out Norwegian auxiliary verbs on the basis of their lemma
    """
    return graph.lemma(node) in NO_AUX_LEMMA

def filter_no_function_words(graph, node):
    """
    filter out Norwegian function words on the basis of the tagger's POS tag
    """
    return graph.pos(node) not in NO_CONTENT_POS


# Greek

GR_CONTENT_POS = set("No Vb Ad Aj".split())

def filter_gr_function_words(graph, node):
    """
    filter out Greek function words on the basis of the tagger's POS tag
    """
    return graph.pos(node)[:2] not in GR_CONTENT_POS

