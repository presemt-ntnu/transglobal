#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate graphs combining nb-2 and bounds scores
"""



import numpy as np
import matplotlib.pyplot as plt

from tg.config import config

nb = np.load("_nb-2.npy")
bounds = np.load("../bounds/_bounds.npy")


for measure in "nist", "bleu", "accuracy":
    fig = plt.figure(figsize=(16,9))
    i = 0
    
    for data in config["eval"]["data_sets"]:
        for lang_pair in config["eval"][data].keys():
            source, target = lang_pair.split("-")
            subset = nb[(nb["data"] == data) & 
                        (nb["source"] == source) & 
                        (nb["target"] == target)]
            
            if not subset.size:
                continue
                    
            i += 1
            
            scores = subset[measure]
            labels = ["class_weigh={}_vect={}".format(r["class_weighting"], 
                                                      r["vect_score_attr"]) 
                      for r in subset]
            
            if measure != "accuracy":
                subset = bounds[(bounds["data"] == data) & 
                                (bounds["source"] == source) & 
                                (bounds["target"] == target)]
                scores = np.hstack([scores, subset[measure]])
                labels += [r["score_attr"] for r in subset]  
                
            ax = plt.subplot(3, 4, i)                       
            
            if measure == "nist":
                plt.xlabel("NIST")
                plt.xlim(xmax=9.0)
            elif measure == "bleu":
                scores *= 100
                plt.xlabel("BLEU (%)")
                plt.xlim(xmax=40.0)
            elif measure == "accuracy":
                scores *= 100
                plt.xlabel("Accuracy (%)")   
                plt.xlim(xmax=100.0)            
            
            y_pos = np.arange(len(labels))    

            handles = plt.barh(y_pos, scores, align='center', height=0.5, 
                     color=("black", "grey", "yellow", "blue",
                            "purple", "brown", "green", "red", "orange"),
                     alpha=0.4)
            
            scores = ["{:.2f}".format(s) for s in scores]
            plt.yticks(y_pos, scores)
            
            plt.grid(axis="x")                
            plt.title("{} {}-{}".format(data.upper(), 
                                        source.capitalize(), 
                                        target.capitalize())) 
            
    plt.figlegend(reversed(handles), reversed(labels), "lower center")
    plt.tight_layout()
    #plt.show()
    plt.savefig("_barchart_" + measure + ".pdf", format="pdf")