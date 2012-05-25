#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
quick hack to write results pickle to table
"""


import sys
import numpy
import numpy as np

for fname in sys.argv[1:]:
    m = np.load(fname)
    np.savetxt(sys.stdout, m, fmt=len(m[1])*"%s\t")
    