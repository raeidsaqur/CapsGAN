#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv
from shutil import copy
import re, os, sys, time
import string

_author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
