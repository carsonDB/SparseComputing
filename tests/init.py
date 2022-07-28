from typing import Optional, List, Tuple, Union
import os
import sys
from torch.nn import functional as F
import torch
import timeit
import numpy as np
import unittest
from torch.testing import assert_close
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.python import *
import sparseOps_cpp as spCpp

torch.set_default_dtype(torch.double)