"""
print some statistics on translation ambiguity in dictionary
"""


from cPickle import load
from tg.config import config
from tg.transdict import ambig_dist, ambig_dist_report

   
pkl_fname = config["dict"]["de-en"]["pkl_fname"]
print "dictionary:", pkl_fname
print "lempos entries only - skipping lemma entries"
trans_dict = load(open(pkl_fname))    
dist = ambig_dist(trans_dict, with_lemma=False)
# for MWUs only:
# dist = ambig_dist(trans_dict, with_lemma=False, with_single_word=False)
# for non-MWUs only:
# dist = ambig_dist(trans_dict, with_lemma=False, with_multi_word=False)
ambig_dist_report(dist)
print "\n"

pkl_fname = config["dict"]["en-de"]["pkl_fname"]
print "dictionary:", pkl_fname
print "lempos entries only - skipping lemma entries"
trans_dict = load(open(pkl_fname))    
dist = ambig_dist(trans_dict, with_lemma=False)
# dist = ambig_dist(trans_dict, with_lemma=False, with_single_word=False)
# dist = ambig_dist(trans_dict, with_lemma=False, with_multi_word=False)
ambig_dist_report(dist)
print "\n"


            
            
        
    
    
    

