import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from tg.config import config


def analyse_fs_2(measure):
    plt.rcParams["axes.color_cycle"] = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey']
    results = np.load("_fs-2-acc.npy")
    
    fig = plt.figure(figsize=(14,14))
    i = 1
    
    for data in "metis", "presemt-dev":
        for lang in config["eval"][data].keys():
            source, target = lang.split("-")
            plt.subplot(4, 2, i)   
               
            for max_freq in [1.0, 0.5, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005]: 
                subset = results[(results["data"] == data) & 
                                 (results["source"] == source) &
                                 (results["target"] == target) &
                                 (results["max_freq"] == max_freq)]
                label="Max Freq = {}".format(max_freq)
                plt.plot(
                    subset["min_count"], 
                    subset[measure] * 100, 
                    "o-",
                    label=label,
                )  
            
            plt.xscale('log')   
            if measure == "accuracy":
                plt.yticks(np.arange(20,70, 5))
                plt.ylim(20, 70) 
            else:
                plt.yticks(np.arange(5,25, 5))
                plt.ylim(5, 25) 
            plt.xlabel("Min Count")   
            plt.ylabel(measure.capitalize() + " (%)")  
            plt.grid(which="both", axis="both" )
            plt.title(data + " " + lang)
            i += 1
        
    plt.legend(fontsize="small", ncol=3)            
    plt.tight_layout()   
    plt.show()
                
            
            
if __name__ == "__main__":
    analyse_fs_2("accuracy")
    analyse_fs_2("bleu")