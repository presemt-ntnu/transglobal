"""
processing of evaluation data
"""

import codecs
import logging
import subprocess
import sys
import xml.etree.ElementTree as et

from tg.config import config


log = logging.getLogger(__name__)


def lemmatize(in_fname, tagger_command, encoding, outf=sys.stdout):
    """
    lemmatize reference translations in mteval format (as used for Presemt
    de-en and en-de evaluation data)
    
    in_fname: input file in mteval format, utf-8 encoded
    
    tagger-command: tree tagger command (e.g. "tree-tagger-english")
    
    encoding: char encoding for *tagger* input (e.g. "latin1" for German
    tagger with latin1 parameter files)

    outf = file or filename for output, utf-8 encoded
    """
    log.info("reading evaluation data from file " + in_fname)
    text =  codecs.open(in_fname, encoding="utf-8").read()
    log.debug("TreeTagger input:\n" + text)
    
    # convert from unicode to char encoding required by tagger
    # we may loose some data here!
    text = text.encode(encoding, "backslashreplace")
    
    # create pipe to tagger
    log.debug("Calling TreeTagger as " + tagger_command)
    tagger_proc = subprocess.Popen(tagger_command, shell=True,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    
    # send text and retrieve tagger output 
    tagger_out, tagger_err = tagger_proc.communicate(text)
    
    # and convert back to unicode
    tagger_out = tagger_out.decode(encoding)        
    log.debug("TreeTagger standard output:\n" + tagger_out)
    log.debug("TreeTagger standard error:\n" + tagger_err)

    if not tagger_out.startswith("<?xml"):
        # prepend char encoding declaration, otherwise xml parser assumes ascii
        tagger_out = u'<?xml version="1.0" encoding="utf-8"?>\n' + tagger_out

    tagger_out = tagger_out.encode("utf-8")
    
    # replace <unknown> tag, because it's invalid xml
    tagger_out = tagger_out.replace("<unknown>", "_unknown_")

    # parse xml from string
    root_elem = et.XML(tagger_out)
    
    # remove word and tag columns
    for seg_elem in root_elem.getiterator("seg"):
        lemmas = []
        
        for line in seg_elem.text.strip().split("\n"):
            # fix: TreeTagger for English sometimes produces output like
            # 'that\t\tIN\tthat',
            line = line.replace("\t\t", "\t")
            word, tag, lemma = line.split("\t")
            
            # if lemma is unknown, default to word form
            if lemma == "_unknown_":
                log.warn("unknown lemma for word " + word)
                lemma = word
                
            lemmas.append(lemma)
    
        seg_elem.text = " ".join(lemmas)
        
    tree = et.ElementTree(root_elem)  
    
    if outf:
        log.info("writing lemmatized evaluation data to {0}".format(
            getattr(outf, "name", outf)))
        tree.write(outf, encoding="utf-8", 
                   xml_declaration='<?xml version="1.0" encoding="utf-8"?>\n')
    return tree
    
    
    
def mteval(ref_fname, src_fname, tst_fname, outf=sys.stdout, options=config["eval"]["mteval_opts"]):
    """
    simple wrapper of NIST mteval-v13a.pl script
    """
    command = "{0} {1} -r {2} -s {3} -t {4} {5}".format(
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
    
    log.debug("mteval standard output:\n" + out)
    
    if err:
        log.error("mteval standard error:\n" + err)
    
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

