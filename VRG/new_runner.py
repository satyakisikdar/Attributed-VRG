import logging
import math
import os
import sys;
from glob import glob
from os.path import join


sys.path.extend(['~/tmp/Attributed-VRG-tmp', '~/tmp/Attributed-VRG-tmp/VRG', '../', '../../'])

from VRG.src import LightMultiGraph
from VRG.src.Tree import create_tree, draw_tree, readjust_tree, tree_okay
from VRG.src.extract import VRGExtractor
from VRG.src.other_graph_models import cell, netgan, dc_sbm, agm_fcl_runner, get_graphs_from_models
from VRG.src.utils import nx_to_lmg, load_pickle, dump_pickle, get_graph_from_prob_matrix

from pathlib import Path
from time import time
from typing import Any, List, Union, Dict, Tuple

import networkx as nx
import seaborn as sns
from tqdm import tqdm
from anytree import RenderTree

from VRG.runner import get_graph, get_clustering, make_dirs, get_grammars

sys.setrecursionlimit(1_000_000)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
# logging.basicConfig(level=logging.ERROR, format="%(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True
sns.set_style('white')


def get_machine_name_and_outdir():
    name, outdir = None, None

    name_path = Path.home().joinpath('.name')
    outdir_path = Path.home().joinpath('.vrg_path')

    if name_path.exists():
        with open(name_path) as fp:
            name = fp.readline().strip()

    if outdir_path.exists():
        with open(outdir_path) as fp:
            outdir = fp.readline().strip()
    return name, outdir


def make_dirs(outdir: str, name: str) -> None:
    """
    Make the necessary directories
    :param outdir:
    :param name:
    :return:
    """
    subdirs = ('grammars', 'graphs', 'trees', 'generators')

    for dir in subdirs:
        dir_path = os.path.join(outdir, dir)
        if not os.path.exists(dir_path):
            logging.error(f'Making directory: {dir_path}')
            os.makedirs(dir_path)
        dir_path = os.path.join(dir_path, name)
        if not os.path.exists(dir_path):
            logging.error(f'Making directory: {dir_path}')
            os.makedirs(dir_path, exist_ok=True)
    return


def main():
    machine_name, outdir = get_machine_name_and_outdir()
    names = ['karate', 'football', 'polbooks', 'wisconsin', 'texas', 'film', 'cornell',
             'cora', 'citeseer', 'airports', 'polblogs', 'chameleon', 'pubmed', 'squirrel']

    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random',
                   'leadingeig', 'consensus'][: -1]

    for name in names:
        g, _ = get_graph(name, basedir=outdir)
        make_dirs(outdir=outdir, name=name)
        for clustering in clusterings:
            tree = load_pickle(join(outdir, 'output', 'trees', name, f'{clustering}_list.pkl'))
            if tree is None: continue
            root = create_tree(tree)
            faulty_tnodes = tree_okay(root=root, g=g)
            if faulty_tnodes > 0: print(f'{name}\t{clustering}\t{faulty_tnodes:,d} errors')


    # name = 'karate'
    # clustering = 'leiden'
    #
    # make_dirs(name=name, outdir=join(outdir, 'output'))
    #
    # input_graph, attr_name = get_graph(gname=name, outdir=outdir)  # this works
    # root = get_clustering(g=input_graph, outdir=outdir, clustering=clustering, use_pickle=True)
    # if isinstance(root, list): root = create_tree(root)
    # draw_tree(root)
    #
    # new_root = readjust_tree(root=root, input_graph=input_graph)

    # grammar_type = 'VRG', 'mu_random'
    # mu = 3
    # lmg: LightMultiGraph = nx_to_lmg(nx_g=input_graph)
    # extractor = VRGExtractor(g=lmg, type=grammar_type[1], mu=mu, root=root, clustering=clustering)
    #
    # grammar = extractor.extract()

    return


def netgan_cell_runner(outdir, name, model):
    from src.other_graph_models import netgan
    input_g, _ = get_graph(name, basedir=outdir)
    print(f'Running {model!r} on {name!r}')
    if model == 'netgan':
        netgan_graphs = netgan(input_g=input_g, name=name, outdir=outdir, use_model_pickle=False)
    else:
        cell_graphs = cell(input_g=input_g, name=name, outdir=outdir, use_model_pickle=False)
    return


def autoencoders(outdir, name, model='gcn_ae'):
    model_path = join(outdir, 'output', 'other_models', 'autoencoders')
    if not Path(model_path).exists():
        os.makedirs(model_path)
    model_path = join(model_path, f'{name}_{model}_mat.pkl')

    if Path(model_path).exists():
        thresh_mat = load_pickle(model_path)
        graphs = [get_graph_from_prob_matrix(thresh_mat, thresh=0.5) for _ in range(10)]
        graphs_path = join(outdir, 'output', 'graphs', name, f'{model}_{len(graphs)}.pkl')
        dump_pickle(graphs, graphs_path)
        return graphs

    from other_models.autoencoders.fit import fit_model
    input_g, _ = get_graph(name, basedir=outdir)
    _, thresh_mat = fit_model(g=input_g, model_name=model)

    dump_pickle(thresh_mat, model_path)
    return


if __name__ == '__main__':
    outdir = '/data/ssikdar/Attributed-VRG'
    names = ['karate', 'football', 'polbooks', 'wisconsin', 'texas', 'cornell', 'airports',
             'polblogs', 'cora', 'citeseer', 'chameleon', 'film', 'pubmed', 'squirrel'][1: ]
    # models = ['netgan', 'cell']
    models = ['gcn_ae', 'gcn_vae', 'linear_ae', 'linear_vae']
    for name in names:
        make_dirs(outdir=join(outdir, 'output'), name=name)
        for model in models:
            # try:
            if model in ('netgan', 'cell'):
                netgan_cell_runner(outdir=outdir, model=model, name=name)
            elif 'ae' in model:
                autoencoders(outdir=outdir, name=name, model=model)
            # except Exception as e:
            #     print(e)
    exit(0)

    for name in names[: ]:
        input_graph, attr_name = get_graph(name, basedir=outdir)
        for model in 'SBM', 'DC-SBM', 'CL', 'AGM':
            try:
                get_graphs_from_models(input_graph=input_graph, num_graphs=10, name=name, model=model, outdir=outdir)
            except Exception as e:
                print(e)

    exit(0)
    #
    # name = 'polblogs'
    # clustering = 'leiden'
    # grammar_type = f'AVRG-greedy-0'
    # mu = 5
    # num_graphs = 10
    #
    # grammar = load_pickle(glob(f'{outdir}/output/grammars/{name}/*AVRG*')[0])
    # graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{clustering}_{mu}_{num_graphs}.pkl'
    # args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
    #              inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))
    #
    # exit(1)

    # autoencoders(outdir=outdir, name=name)
    # for name in names[1: -4]:
    #     netgan_cell_runner(outdir, name)
    # exit(1)

    # main()
    # Extracting grammar name:lang-bip mu:3 type:all_tnodes clustering:leiden
    # mu = 3
    # grammar_type = 'AVRG', 'all_tnodes'
    # clustering = 'leiden'
    # input_graph, attr_name = get_graph(name, basedir=outdir)
    # print(input_graph)
    # vrg = get_grammars(name=name, attr_name=attr_name, clustering=clustering, grammar_type=grammar_type,
    #                    input_graph=input_graph, mu=mu, use_cluster_pickle=True, use_grammar_pickle=True,
    #                    outdir=outdir, )
    # print(vrg)
