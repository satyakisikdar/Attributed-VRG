"""
Use the Matlab implementation to find the consensus hierarchical clustering of a network
"""
import logging
import os
import pickle
import platform
import sys;
from collections import deque
from pathlib import Path

sys.path.extend(['./..', '../', '../../'])
import numpy as np
import networkx as nx
import subprocess as sub

import matplotlib

from VRG.src.Tree import TreeNode
matplotlib.use('Qt5agg')


def check_file_exists(path) -> bool:
    """
    Checks if file exists at path
    :param path:
    :return:
    """
    if isinstance(path, str):
        path = Path(path)
    return path.exists()


def run_matlab_code(g, gname):
    matlab_code_path = './src/matlab_clustering/HierarchicalConsensus/'
    adj_mat_path = os.path.join(matlab_code_path, f'data/{gname}.mat')

    if not check_file_exists(adj_mat_path):
        np.savetxt(adj_mat_path, nx.to_numpy_matrix(g), fmt='%d')

    matlab_code_filename = os.path.join(matlab_code_path, f'{gname}_code.m')

    matlab_code = [
        'addpath(genpath(\'./src/matlab_clustering\'));',
        'cd src/matlab_clustering/HierarchicalConsensus;',
        f'A = dlmread(\'./data/{gname}.mat\');',
        'S = exponentialSamples(A, 500);',
        '[Sc, Tree] = hierarchicalConsensus(S);',
        f'dlmwrite("./data/{gname}_sc.vec", Sc, \' \');',
        f'dlmwrite("./data/{gname}_tree.mat", Tree, \' \');'
    ]

    print('\n'.join(matlab_code), file=open(matlab_code_filename, 'w'))

    tree_sc_path = os.path.join(matlab_code_path, f'data/{gname}_tree.mat')
    if not check_file_exists(tree_sc_path):
        if 'Linux' in platform.platform():
            runner_code = f'source ~/.bashrc; cat {matlab_code_filename} | matlab -nosplash_desktop'
            executable = '/bin/bash'
        else:
            runner_code = f'source ~/.zshrc; cat {matlab_code_filename} | matlab -nosplash -nodesktop'
            executable = '/bin/zsh'
        process = sub.run(runner_code, shell=True, executable=executable)
        assert process.returncode == 0
        assert check_file_exists(tree_sc_path)

    return


def get_consensus_root(g, gname):
    run_matlab_code(g, gname)
    clust_arr = np.loadtxt(f'./src/matlab_clustering/HierarchicalConsensus/data/{gname}_sc.vec', dtype=int)

    clustering = {}
    for u, c_u in zip(g.nodes(), clust_arr):
        if c_u not in clustering:
            clustering[c_u] = set()
        clustering[c_u].add(u)

    tree_graph = nx.read_weighted_edgelist(f'./src/matlab_clustering/HierarchicalConsensus/data/{gname}_tree.mat',
                                           nodetype=int)

    root = TreeNode(name='n1')
    stack = deque([(1, root)])
    visited = {1}

    while len(stack) != 0:
        u, tnode_u = stack.popleft()
        if u in clustering:
            for leaf in clustering[u]:
                tnode_l = TreeNode(name=leaf, parent=tnode_u)
        for v in tree_graph.neighbors(u):
            if v not in visited:
                visited.add(v)
                tnode_v = TreeNode(name=f'n{v}', parent=tnode_u)
                stack.append((v, tnode_v))

    root_pickle_path = f'./dumps/trees/{gname}/consensus.pkl'
    logging.error(f'Dumping root pickle at {root_pickle_path!r}')
    pickle.dump(root, open(root_pickle_path, 'wb'))

    return root


def main():
    g = nx.karate_club_graph(); gname = 'karate'
    get_consensus_root(g=g, gname=gname)
    return


if __name__ == '__main__':
    main()
