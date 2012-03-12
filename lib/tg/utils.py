import logging
import codecs

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
        
        
def set_default_log(level=logging.INFO):
    """
    configure default "root" logger
    """ 
    # get root logger
    log = logging.getLogger()
    # output to stderr
    log_handler = logging.StreamHandler()
    log_format = logging.Formatter('| %(levelname)-8s | %(name)-24s | %(message)s')
    log_handler.setFormatter(log_format)
    log.addHandler(log_handler)
    log.setLevel(level)
    return log
    
    
        

