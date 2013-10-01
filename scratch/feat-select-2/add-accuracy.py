#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ad-hoc script to add accuracy 
"""

import cPickle

import numpy as np
import asciitable as at

from tg.config import config
from tg.accuracy import accuracy_score
from tg.utils import set_default_log

set_default_log()

#name = "fs-1"
name = "fs-2"    

old_results = np.load("_" + name + ".npy")


descriptor = [ 
        ("data", "S16"),
        ("source", "S8"),
        ("target", "S8"),
        ("min_count", "f"),
        ("max_freq", "f"),
        ("correct", "i"),
        ("incorrect", "i"),
        ("ignored", "i"),
        ("accuracy", "f"),
        ("graphs", "i"),
        ("nist", "f"),
        ("bleu", "f"),
        ("exp_name", "S128"),   
    ] 

new_results = np.zeros(len(old_results), descriptor)

for i, exp in enumerate(old_results):
    ref_fname = config["eval"][exp["data"]][exp["source"] + "-" + exp["target"]]["lemma_ref_fname"]
    graphs_fname = "_{}/{}_graphs.pkl".format(name, exp["exp_name"])
    graphs = cPickle.load(open(graphs_fname))
    accuracy = accuracy_score(graphs, ref_fname, name + "_score")
    new_results[i]["graphs"] = len(graphs)
    new_results[i]["data"] = exp["data"]
    new_results[i]["source"] = exp["source"]
    new_results[i]["target"] = exp["target"]
    new_results[i]["min_count"] = exp["min_count"]
    new_results[i]["max_freq"] = exp["max_freq"]
    new_results[i]["correct"] = accuracy.correct
    new_results[i]["incorrect"] = accuracy.incorrect
    new_results[i]["accuracy"] = accuracy.score
    new_results[i]["ignored"] = accuracy.ignored
    new_results[i]["nist"] = exp["nist"]
    new_results[i]["bleu"] = exp["bleu"]
    new_results[i]["exp_name"] = exp["exp_name"]

np.save("_" + name + "-acc.npy", new_results)

at.write(new_results, 
         "_" + name + "-acc.txt", 
         Writer=at.FixedWidthTwoLine, 
         delimiter_pad=" ")

    
    