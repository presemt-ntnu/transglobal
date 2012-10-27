"""
parse NIST and BLEU scores from output of mteval scoring script
"""

import codecs
import logging
import re
import subprocess
import sys

import numpy as np

from tg.config import config

log = logging.getLogger(__name__)

  
    
def mteval(ref_fname, src_fname, tst_fname, outf=sys.stdout,
           options=config["eval"]["mteval_opts"]):
    """
    Simple wrapper of NIST mteval-v13a.pl script
    
    Parameters
    ----------
    ref_fname: str
        file containing the reference translations for the documents
        to be evaluated
    src_fname: str
        file containing the source documents for which translations 
        are to be evaluated 
    tst_fname: str
        file containing the translations to be evaluated
    outf: file
        output file
    options: str
        eval script command line options
    
    Returns
    -------
    out, err: str, str
        strings received from standard output and error
    """
    command = '{0} {1} -r "{2}" -s "{3}" -t "{4}" {5}'.format(
        config["eval"]["perl_fname"],
        config["eval"]["mteval_fname"],
        ref_fname,
        src_fname,
        tst_fname,
        options or "")
        
    # create pipe to eval script
    log.debug("Calling evaluation script as as: " + command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
      
    # send text and retrieve tagger output 
    out, err = proc.communicate()    
    
    log.debug(u"mteval standard output:\n" + out.decode("utf-8"))
    
    if err:
        log.error(u"mteval standard error:\n" + err.decode("utf-8"))
    
    if isinstance(outf, basestring):
        outf = open(outf, "w")
        close = True
    else:
        close = False
        
    log.info("writing mteval output to " + outf.name)
    outf.write(out)
    
    if close:
        outf.close()
    
    return out, err



def parse_total_scores(file):
    """
    Get NIST and BLEU scores from score file
    
    Parameters
    ----------
    file: file or filename
        Output from mteval-v13a.pl scoring script
        produced with flag -d2 or -d3 for detailed output
        
    Returns
    -------
    (system, NIST, BLEU): tuple
        Tuple of system identifier and scores
    """
    if isinstance(file, basestring):
        file = open(file)
        
    total_score_pat = re.compile(
        'NIST score = ([0-9.]+)\s+BLEU score = ([0-9.]+)\s+for system "(.*)"')
    
    for line in file:
        try:
            nist, bleu, system = total_score_pat.match(line).groups()
        except AttributeError:
            pass
        else:
            return system, float(nist), float(bleu)
        
        
def parse_document_scores(file):
    """
    Get NIST and BLEU scores for documents from score file
    
    Parameters
    ----------
    file: file or filename
        Output from mteval-v13a.pl scoring script
        produced with flag -d2 or -d3 for detailed output 
    Returns
    -------
    scores: numpy.ndarray
        array with named fields "system", "document", "segments", "words", 
        "NIST" and "BLEU"
    """
    if isinstance(file, basestring):
        file = open(file)
        
    src_set_pat = re.compile(    
        '\s+src set ".*" \(([0-9.]+) docs, \d+ segs\)')        

    doc_score_pat = re.compile(
        '(NIST|BLEU) score using\s+\d+-grams = ([0-9.]+) for system "(.*)" '
        'on document "(.*)" \((\d+) segments, (\d+) words\)')  
    
    descriptor = {"names": ("system", "document", "segments", "words", 
                            "NIST", "BLEU", ), 
                  "formats": ("S64", "S64", "i4", "i4", "f4", "f4")}  
    
    doc_score_count = 0
    
    for line in file:
        try:
            n_docs = int(src_set_pat.match(line).group(1))
        except AttributeError:
            pass
        else:
            scores = np.zeros(n_docs, dtype=descriptor)
        
        try:
            metric, score, system, document, segments, words =\
            doc_score_pat.match(line).groups()
        except AttributeError:
            pass
        else:
            row = scores[doc_score_count % n_docs]
            row["system"] = system
            row["document"] = document
            row["segments"] = int(segments)
            row["words"] = int(words)
            row[metric] = float(score)
            doc_score_count += 1
            
    return scores


def parse_segment_scores(file):
    """
    Get NIST and BLEU scores for segments (sentences) from score file
    
    Parameters
    ----------
    file: file or filename
        Output from mteval-v13a.pl scoring script
        produced with flag -d2 or -d3 for detailed output
        
    Returns
    -------
    scores: numpy.ndarray
        array with named fields "system", "document", "segment", "words", 
        "NIST" and "BLEU"
    """
    if isinstance(file, basestring):
        file = open(file)
        
    src_set_pat = re.compile(    
        '\s+src set ".*" \(\d+ docs, ([0-9.]+) segs\)')        
    
    sent_score_pat = re.compile(
        '  (NIST|BLEU) score using \d+-grams = ([0-9.]+) for system "(.*)" '
        'on segment (\d+) of document "(.*)" \((\d+) words\)')
    
    descriptor = {"names": ("system", "document", "segment", "words", 
                            "NIST", "BLEU", ), 
                  "formats": ("S64", "S64", "S64", "i4", "f4", "f4")}  
    
    seg_score_count = 0
    
    for line in file:
        try:
            n_seg = int(src_set_pat.match(line).group(1))
        except AttributeError:
            pass
        else:
            scores = np.zeros(n_seg, dtype=descriptor)
        
        try:
            metric, score, system, segment, document, words =\
            sent_score_pat.match(line).groups()
        except AttributeError:
            pass
        else:
            row = scores[seg_score_count % n_seg]
            row["system"] = system
            row["document"] = document
            row["segment"] = segment
            row["words"] = int(words)
            row[metric] = float(score)
            seg_score_count += 1 
            
    return scores

