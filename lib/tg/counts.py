#!/usr/bin/env python

"""
pickle corpus counts

Pickled dict is used for calculating "most frequent" baseline.

Examples of how to produce sorted lemma and lempos counts using the Manatee
command line tool 'lslex' (assuming Bash shell):

lsclex -f deTenTen lemma | sort -n -r -t $'\\t' -k 3 | bzip2 > deTenTen_id-lemma-count.bz2

lsclex -f deTenTen lempos | sort -n -r -t $'\\t' -k 3 | bzip2 > deTenTen_id-lempos-count.bz2

The output file contains three tab-delimited fields: 
(1) lemma id; (2) lemma or lempos; (3) corpus count.
Character encoding is assumed to be utf-8.
"""

import bz2
import codecs
import cPickle
import logging


log = logging.getLogger(__name__)


def mk_counts_pkl(counts_fname, pkl_fname, min_count=1):
    log.info("reading counts from " + counts_fname)
    log.info("min_count = {0}".format(min_count)) 
    
    if counts_fname.endswith("bz2"):
        inf = bz2.BZ2File(counts_fname)
    else:
        # decode line explicitly rather than using codecs.open to wrap the
        # file, because otherwise there is no way to resume after a
        # UnicodeDecodeError due to invalid bytes
        inf = open(counts_fname)
        
    # token can be lempos, lemma, etc
    counts = {}
    
    for line in inf:
        # decode line explicitly rather than using codecs.open to wrap the
        # file, because otherwise there is no way to resume after a
        # UnicodeDecodeError due to invalid bytes
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        
        try:
            id, token, count = line.split("\t")
        except ValueError:
            # skip ill-formed line
            continue
        
        count = int(count)
        
        if count <= min_count:
            # remaining tokens too infrequent
            break
        
        log.debug(u"Adding: {0}".format(token))
        counts[token] = count
            
    log.info("counts dict size = {0}".format(len(counts))) 
    log.info("saving counts dict to " + pkl_fname)
    with open(pkl_fname, "wb") as f:
        cPickle.dump(counts, f)
        