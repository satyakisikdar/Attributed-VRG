"""
Class for a hierarchical clustering algorithm
"""
import logging
import time
from os.path import join
from pathlib import Path
from statistics import mean, median
from typing import Union, Dict

import networkx as nx
import igraph as ig
from dataclasses import dataclass, field

import math
from anytree import LevelOrderIter

from VRG.src.Tree import TreeNode, create_tree, dasgupta_cost
from VRG.src.partitions import louvain_leiden_infomap_label_prop, get_random_partition, \
    spectral_kmeans, approx_min_conductance_partitioning
from VRG.src.utils import load_pickle, dump_pickle


@dataclass
class HierarchicalClustering:
    """
    Base class for hierarchical clustering algorithms
    """
    input_nx_graph: nx.Graph
    name: str
    clustering: str
    root: Union[TreeNode, None] = None
    root_pickle_filename: str = ''
    basedir: str = '/data/ssikdar/Attributed-VRG'

    def __post_init__(self):
        if self.root_pickle_filename == '':
            self.root_pickle_filename = join(self.basedir, 'output', 'trees', self.name, f'{self.clustering}_root.pkl')
        self.stats = dict(name=self.name, clustering=self.clustering)
        return

    def get_clustering(self, use_pickle: bool = False):
        """
        runs the clustering algorithm and returns the root of the TreeNode
        :return:
        """
        if use_pickle and Path(self.root_pickle_filename).exists():
            pickled_root = load_pickle(self.root_pickle_filename)
            if pickled_root is None:
                logging.error(f'Error in pickle! {self.name!r} {self.clustering!r}')
            else:
                logging.error(f'Using pickled root {self.name!r} {self.clustering!r}!')
                self.root = pickled_root
                self.stats.update(self.get_tree_stats())
                return

        list_of_list_clusters = []
        start_time = time.perf_counter()

        if self.clustering in ('leiden', 'louvain', 'labelprop', 'infomap', 'leideneig'):
            list_of_list_clusters = louvain_leiden_infomap_label_prop(g=self.input_nx_graph, method=self.clustering)
        elif self.clustering == 'random':
            list_of_list_clusters = get_random_partition(g=self.input_nx_graph)
        elif self.clustering == 'spectral':
            K = int(math.sqrt(self.input_nx_graph.order()//2))
            list_of_list_clusters = spectral_kmeans(g=self.input_nx_graph, K=K)
        elif self.clustering == 'cond':
            list_of_list_clusters = approx_min_conductance_partitioning(g=self.input_nx_graph)
        elif self.clustering == 'hyphc':
            raise NotImplementedError()

        self.root = create_tree(list_of_list_clusters)
        end_time = time.perf_counter()
        dump_pickle(self.root, self.root_pickle_filename)

        logging.error(f'Running {self.clustering!r} on {self.name!r} took {(end_time - start_time):.2g} secs')
        self.stats.update(self.get_tree_stats())
        logging.error(self.stats)
        return

    def get_tree_stats(self, fast=True) -> Dict:
        """
        Compute height of the tree, avg branching factor, dasgupta cost
        """
        ht = self.root.height
        branch_factors = [len(node.children) for node in LevelOrderIter(self.root) if len(node.children) > 1]
        avg_branch_factor = mean(branch_factors)
        median_branch_factor = median(branch_factors)
        dc = -1 if fast else dasgupta_cost(g=self.input_nx_graph, root=self.root)

        stats = dict(height=ht, avg_branch_factor=round(avg_branch_factor, 3),
                     median_branch_factor=round(median_branch_factor, 3), cost=dc)
        return stats
