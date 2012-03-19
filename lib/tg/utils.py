import logging
import codecs
import sys

from os import makedirs
from os.path import exists, isfile, dirname


def create_dirs(path):
    """
    create directores if they do not exists yet;
    path is assumed to be a path to a *file* (not a dir)
    """
    path = dirname(path)        
    if not exists(path):
        makedirs(path)
        
        
def set_default_log(level=logging.INFO, encoding="utf-8", errors="strict"):
    """
    configure default "root" logger
    """ 
    # get root logger
    log = logging.getLogger()
    # output to wrapped stderr
    stream = codecs.getwriter(encoding)(sys.stderr, errors=errors)
    log_handler = logging.StreamHandler(stream)
    log_format = logging.Formatter('| %(levelname)-8s | %(name)-24s | %(message)s')
    log_handler.setFormatter(log_format)
    log.addHandler(log_handler)
    log.setLevel(level)
    return log


def indent(elem, level=0):
    """
    elementree formatting: indent element and all subelements for pretty
    printing of XML
    """
    # copied from Fredrik Lund 
    # http://effbot.python-hosting.com/file/effbotlib/ElementTree.py
    i = "\n" + level*"  "
    
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i

    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

    
    
        

