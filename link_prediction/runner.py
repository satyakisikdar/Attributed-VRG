import logging
from typing import Tuple, List

import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

from methods.VRG.runner import get_grammars, get_graph, make_dirs
from methods.VRG.src.VRG import NCE
from methods.VRG.src.generate import GreedyGenerator, NCEGenerator, EnsureAllNodesGenerator
from methods.VRG.src.utils import check_file_exists, load_pickle, dump_pickle
from methods.autoencoders.linear_gae.gae_fit import fit_model
from utils import sparse_to_tuple, make_plot, sigmoid


