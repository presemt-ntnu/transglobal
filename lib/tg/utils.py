import codecs

from os import makedirs
from os.path import exists, isfile, dirname


def create_dirs(path):
    """
    create directores if they do not exists yet
    """
    if isfile(path):
        path = dirname(path)
        
    if not exists(path):
        makedirs(path)
        

