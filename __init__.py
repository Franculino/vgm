"""
Vascular Graph Model (VGM)
==========================

VGM is a package for the simulation of cerebral blood flow in realistic
vascular networks. It makes use of the graph package iGraph to implement the
graph structure of the vasculature. Numpy and scipy are used extensively for
numerical manipulations. An algebraic multigrid solver is provided by PyAMG.
Matplotlib is used for 2D plots. 
Please consult the accompanying sphinx documentation for installation
instructions and tutorials. Moreover, the docstrings of modules, classes, and
functions should be helpful in understanding the code.
"""

__author__   = 'Johannes Reichold'

# Read configuration file and make it available to all vgm modules via
# vmg.ConfParser:
import ConfigParser
import os
basedir = __path__[-1]
ConfParser = ConfigParser.SafeConfigParser()
ConfParser.read(os.path.join(basedir, 'conf/vgm.conf'))

# Prepare logging by creating a LoggingDispatcher:
import logger
LogDispatcher = logger.LoggingDispatcher()


# VGM imports:
from core import *
#from preprocessing import *
import core
#import preprocessing


