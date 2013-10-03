"""
generic framework for running translation experiments with classifiers 
"""

import cPickle
import logging
import os
import tempfile

import h5py

from tg.config import config
from tg.exps import support

# Functions and classes below are imported in local namespace 
# and will therefore become part of exp's namespace'
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator
from tg.classify import TranslationClassifier
from tg.model import ModelBuilder
from tg.classcore import ClassifierScore, Vectorizer, filter_functions
from tg.bestscore import BestScorer
from tg.draw import Draw
from tg.format import TextFormat, MtevalFormat
from tg.mteval import mteval, parse_total_scores
from tg.transdiff import trans_diff
from tg.accuracy import accuracy_score


log = logging.getLogger(__name__)

# dummy function used for skipping certain steps
SKIP = lambda *args, **kwargs: None


#-------------------------------------------------------------------------------
# Setup experiment
#-------------------------------------------------------------------------------

def setup(ns):
    ns.make_exp_dir(ns)
    ns.create_filename_prefix(ns)
    ns.get_languages(ns)
    ns.get_graphs(ns)
    
def make_exp_dir(ns):  
    ns.exp_dir = "_" + ns.name
    if not os.path.exists(ns.exp_dir):
        log.info("creating exp dir " + ns.exp_dir)
        os.makedirs(ns.exp_dir)
    
def create_filename_prefix(ns):
    ns.fname_prefix = tempfile.NamedTemporaryFile(
        dir=ns.exp_dir,
        prefix=ns.name + "_").name
    ns.exp_name = os.path.basename(ns.fname_prefix)

def get_graphs(ns):
    ns.graphs_fname = config["eval"][ns.data][ns.lang]["graphs_fname"]    
    log.info("loading graphs from " + ns.graphs_fname) 
    ns.graphs = cPickle.load(open(ns.graphs_fname))[:ns.n_graphs]  
    
def get_languages(ns):
    ns.source_lang, ns.target_lang = ns.lang.split("-")
        

#-------------------------------------------------------------------------------
# Build models
#-------------------------------------------------------------------------------

# This tries to avoid keeping resources in memory by not storing large
# objects such as the AmbguityMap and HDF5 samples in the namespace

def build_models(ns):
    ns.models_fname = ns.fname_prefix + "_models.hdf5"   
    model_builder = ns.ModelBuilder(ns.get_data_generator(ns), 
                                    ns.models_fname,
                                    ns.classifier)
    model_builder.run()
    
def get_data_generator(ns):
    return ns.DataSetGenerator(ns.get_ambiguity_map(ns), 
                               ns.get_samples(ns),
                               shuffle=getattr(ns, "shuffle", False),
                               random_state=getattr(ns, "random_state", None))
            
def get_ambiguity_map(ns):
    ambig_fname = config["sample"][ns.lang]["ambig_fname"]            
    return ns.AmbiguityMap(ambig_fname, graphs=ns.graphs)    
    
def get_samples(ns):
    try:
        ns.samples_fname = config["sample"][ns.lang]["samples_filt_fname"]
    except KeyError:
        ns.samples_fname = config["sample"][ns.lang]["samples_fname"]
        log.warn("Backing off to unfiltered samples from " + 
                 ns.samples_fname) 
        
    log.info("opening samples file " + ns.samples_fname)
    return h5py.File(ns.samples_fname, "r")
    
    
#-------------------------------------------------------------------------------
# Scoring translation candidates
#-------------------------------------------------------------------------------    

def score(ns):
    ns.compute_classifier_score(ns)
    ns.compute_best_score(ns)
    ns.save_scored_graphs(ns)
    
def compute_classifier_score(ns):
    models = ns.TranslationClassifier(ns.models_fname)
    if not ns.score_attr:
        ns.score_attr = ns.name + "_score"
    if not hasattr(ns, "vectorizer"):
        ns.vectorizer = Vectorizer()
    scorer = ns.ClassifierScore(models,
                                score_attr=ns.score_attr,
                                filter=filter_functions(ns.source_lang),
                                vectorizer=ns.vectorizer)
    scorer(ns.graphs)
    
def compute_best_score(ns):
    ns.base_score_attrs = [ns.score_attr, "freq_score"]
    ns.best_score_attr = "best_core"
    best_scorer = ns.BestScorer(base_score_attrs=ns.base_score_attrs,
                                score_attr=ns.best_score_attr)
    best_scorer(ns.graphs)    
    
def save_scored_graphs(ns):
    ns.scored_graphs_fname = ns.fname_prefix + "_graphs.pkl"
    log.info("saving scored graphs to " + ns.scored_graphs_fname)
    cPickle.dump(ns.graphs, 
                 open(ns.scored_graphs_fname, "w"))        
        
        
#-------------------------------------------------------------------------------
# Evaluation
#-------------------------------------------------------------------------------

def evaluate(ns):
    ns.compute_nist_blue_scores(ns)
    ns.compute_accuracy_score(ns)
    
def compute_nist_blue_scores(ns):    
    ns.write_mteval_format(ns)
    ns.run_mteval(ns)
    
def write_mteval_format(ns):
    mte_format = ns.MtevalFormat(
        config["eval"][ns.data][ns.lang]["src_fname"],
        trglang=ns.target_lang, 
        sysid=ns.name,
        score_attr=ns.best_score_attr)
    mte_format(ns.graphs)
    ns.tst_fname = ns.fname_prefix + ".tst"
    mte_format.write(ns.tst_fname)
    
def run_mteval(ns):  
    ns.scores_fname = ns.fname_prefix +  "_mteval.txt"
    mteval(config["eval"][ns.data][ns.lang]["lemma_ref_fname"],
           config["eval"][ns.data][ns.lang]["src_fname"],
           ns.tst_fname,
           ns.scores_fname)
    ns.scores = ns.parse_total_scores(ns.scores_fname)  
    
def compute_accuracy_score(ns):
    # do not compute accuracy on best score , but on model score
    ns.accuracy = ns.accuracy_score(
        ns.graphs,
        config["eval"][ns.data][ns.lang]["lemma_ref_fname"], 
        ns.score_attr)
    

#-------------------------------------------------------------------------------
# Postprocess
#-------------------------------------------------------------------------------
    
def postprocess(ns):
    ns.thrash_models(ns)
    ns.draw_graphs(ns)
    ns.write_diff(ns)
    ns.write_text(ns)
    ns.get_class_params(ns)
    
def thrash_models(ns):    
    log.info("removing models file " + ns.models_fname)
    os.remove(ns.models_fname)  
    ns.models_fname = None    
    
def draw_graphs(ns):
    draw = ns.Draw()
    draw(ns.graphs, out_format="pdf", best_score_attr=ns.best_score_attr,
         base_score_attrs=ns.base_score_attrs, fname_prefix=ns.fname_prefix)
    
def write_diff(ns):
    ref_fname = config["eval"][ns.data][ns.lang]["lemma_ref_fname"]
    ns.diff_fname = ns.fname_prefix + "_diff.txt"
    ns.trans_diff(ns.graphs, ns.base_score_attrs, ref_fname,
                  outf=ns.diff_fname)

def write_text(ns):
    text_format = ns.TextFormat(score_attr=ns.best_score_attr)
    text_format(ns.graphs)
    ns.text_fname = ns.fname_prefix + "_text.txt"
    text_format.write(ns.text_fname)
    
def get_class_params(ns):
    # copy classifier parameters to namespace for more convenient access
    if ns.classifier:
        ns.import_locals(ns.classifier.get_params())
    


#-------------------------------------------------------------------------------
# Single experiment
#-------------------------------------------------------------------------------

@support.grid_search    
def single_exp(name, 
               classifier, 
               data, 
               lang,
               vectorizer=Vectorizer(),
               n_graphs=None,
               score_attr=None,
               thrash_models=SKIP,
               **kwargs):
    """
    Parameters
    ----------
    name
    classifier
    data
    lang
    
    vectorizer
    n_graphs
    score_attr
    exp_dir
    fname_prefix
    
    AmbiguityMap
    ModelBuilder
    TranslationClassifier
    
    shuffle
    
    """
    # Create namespace instance holding all experimental parameters.
    ns = support.Namespace()
    # import functions from experiments module 
    ns.import_module("tg.exps.experiment")
    # overide with values of function's parameters and keyword args
    ns.import_locals(locals())
    
    ns.setup(ns)
    ns.build_models(ns)
    ns.score(ns)
    ns.evaluate(ns)
    ns.postprocess(ns)
    
    return ns    


