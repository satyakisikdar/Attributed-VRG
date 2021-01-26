import networkx as nx
import igraph as ig
import numpy as np
import glob
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns;


sns.set_style('white')
import sys; sys.path.extend(['../', '../../', ])
from VRG.src.generate import RandomGenerator

from time import time
import logging
from anytree import LevelOrderIter
from statistics import mean
import os

from VRG.src.VRG import VRG
from VRG.src.utils import load_pickle, nx_to_igraph, check_file_exists
from VRG.src.graph_stats import GraphStats
from VRG.src.graph_comparison import GraphPairCompare
from VRG.runner import get_graph
from VRG.src.Tree import create_tree, dasgupta_cost
from VRG.src.MDL import graph_dl as graph_mdl


def get_compatibility_matrix(g: nx.Graph, attr_name: str):
    """
    From Danai's heterophily paper
    :param g:
    :param attr_name:
    :return:
    """
    if max(g.nodes) != g.order() - 1:
        g = nx.convert_node_labels_to_integers(g, first_label=0)
    values = set(nx.get_node_attributes(g, attr_name).values())
    mapping = {val: i for i, val in enumerate(values)}
#     print(mapping)
    C = nx.attribute_mixing_matrix(g, attribute=attr_name, mapping=mapping, normalized=False)
    np.fill_diagonal(C, C.diagonal() / 2)

    D = np.diag(np.diag(C))
    e = np.ones(shape=(len(mapping), 1))

    h = float((e.T @ D @ e) / (e.T @ C @ e))

    Y = np.zeros(shape=(g.order(), len(mapping)))
    for n, d in g.nodes(data=True):
        attr = d[attr_name]
        Y[n, mapping[attr]] = 1
    A = nx.adjacency_matrix(g)
    E = np.ones(shape=(A.shape[0], len(mapping)))

    H = (Y.T @ A @ Y) / (Y.T @ A @ E)

    return_d = dict(homophily_ratio=h, compatibility_mat=H, attr_name=attr_name, mapping=mapping)
    return return_d


def get_tree_stats(g, root):
    """
    Compute height of the tree, avg branching factor, dasgupta cost
    """
    ht = root.height
    avg_branch_factor = mean(len(node.children) for node in LevelOrderIter(root) if len(node.children) > 1)
    dc = dasgupta_cost(g, root)
    return ht, avg_branch_factor, dc


def make_graph_df(name, fname, orig_graph, mu, clustering, attr_name, grammar_type, bipartite=False):
    deg_ast_fn = nx.degree_assortativity_coefficient
    attr_ast_fn = nx.attribute_assortativity_coefficient

    gen_graphs = load_pickle(fname)
    if gen_graphs is None:
        return pd.DataFrame()

    cols = ['name', 'orig_n', 'orig_m', 'orig_degree_ast', 'orig_sp_ast_50', 'orig_sp_ast_100', 'orig_sp_ast_500',
            'attr_name', 'orig_attr_ast',
            'model', 'mu', 'clustering', 'gen_n', 'gen_m', 'gen_degree_ast', 'gen_sp_ast_50', 'gen_sp_ast_100', 'gen_sp_ast_500',
            'gen_attr_ast', 'total_rewired_edges', 'fancy_rewired_edges',
            'degree_js', 'pagerank_js', 'lambda_dist', 'deg_ast_diff', 'attr_ast_diff', 'is_bipartite']

    row = {col: np.nan for col in cols}

    orig_deg_ast = deg_ast_fn(orig_graph)

    orig_attr_ast = attr_ast_fn(orig_graph, attr_name) if attr_name != '' else np.nan
    orig_gstats = GraphStats(orig_graph)
    orig_sp_ast_50 = orig_gstats.shortest_path_ast(alpha=0.5)
    orig_sp_ast_100 = orig_gstats.shortest_path_ast(alpha=1)
    orig_sp_ast_500 = orig_gstats.shortest_path_ast(alpha=5)

    orig_h_dict = get_compatibility_matrix(orig_graph, attr_name)
    orig_h = orig_h_dict['homophily_ratio']
    orig_h_mat = orig_h_dict['compatibility_mat']
    orig_h_map = orig_h_dict['mapping']
    orig_is_bip = nx.algorithms.bipartite.is_bipartite(orig_graph) if bipartite else np.nan

    rows = []

    for g in gen_graphs:
        gen_gstats = GraphStats(g)
        gpc = GraphPairCompare(orig_gstats, gen_gstats)
        gen_deg_ast = deg_ast_fn(g)
        gen_sp_ast_50 = gen_gstats.shortest_path_ast(alpha=0.5)
        gen_sp_ast_100 = gen_gstats.shortest_path_ast(alpha=1)
        gen_sp_ast_500 = gen_gstats.shortest_path_ast(alpha=5)

        gen_attr_ast = attr_ast_fn(g, attr_name) if attr_name != '' else np.nan
        total_rewired_edges = g.graph.get('total_rewirings', 0)
        fancy_rewired_edges = g.graph.get('fancy_rewirings', 0)
        h_dict = get_compatibility_matrix(g, attr_name)
        h = h_dict['homophily_ratio']
        h_mat = h_dict['compatibility_mat']
        h_map = h_dict['mapping']

        gen_is_bip = nx.algorithms.bipartite.is_bipartite(g) if bipartite else np.nan

        row = dict(name=name, orig_n=orig_graph.order(), orig_m=orig_graph.size(), orig_deg_ast=orig_deg_ast,
                   orig_sp_ast_50=orig_sp_ast_50, orig_sp_ast_100=orig_sp_ast_100, orig_sp_ast_500=orig_sp_ast_500,
                   orig_attr_ast=orig_attr_ast, attr_name=attr_name, model=grammar_type, clustering=clustering,
                   mu=mu, orig_homophily_ratio=orig_h, orig_homophily_mat=orig_h_mat, orig_homophily_map=orig_h_map,
                   orig_is_bipartite=orig_is_bip,
                   gen_n=g.order(), gen_m=g.size(), gen_deg_ast=gen_deg_ast,
                   gen_sp_ast_50=gen_sp_ast_50, gen_sp_ast_100=gen_sp_ast_100, gen_sp_ast_500=gen_sp_ast_500,
                   gen_attr_ast=gen_attr_ast,
                   total_rewired_edges=total_rewired_edges, fancy_rewired_edges=fancy_rewired_edges,
                   degree_js=gpc.degree_js(), pagerank_js=gpc.pagerank_js(), lambda_dist=gpc.lambda_dist(),
                   gen_homophily_ratio=h, gen_homophily_mat=h_mat, gen_homophily_map=h_map,
                   deg_mix_dist_dict=gpc.deg_mixing_dist_dict(), attr_mix_dist_dict=gpc.attr_mixing_dist_dict(),
                   gen_is_biparite=gen_is_bip)
        rows.append(row)
    return pd.DataFrame(rows)


def make_grammar_df(base_path, names=None, graphs=None):
    if names is None:
        names = ['karate', 'football', 'polbooks', 'us-flights', 'cora', 'citeseer', 'polblogs', 'pubmed']
    root_dict = {f'{name}': {} for name in names}
    dl_dict = {}

    for name in names:
        orig_graph, attr_name = get_graph(name, basedir=base_path)
        dl_dict[f'{name}-plain'] = graph_mdl(orig_graph, attributed=False)
        dl_dict[name] = graph_mdl(orig_graph, attributed=True)
    print(dl_dict)

    rows = []
    for name in names:
        temp_fname = f'{base_path}/stats/_grammar_df_{name}.csv'
        if Path(temp_fname).exists():
            print(f'Skipping {name!r}')
            continue
        orig_graph, attr_name = get_graph(name, basedir=base_path)
        print('\n\n', name, attr_name)
        for fname in glob.glob(f'{base_path}/output/grammars/{name}/*.pkl'):
            path = Path(fname)
            pattern = r'(.+)_(\w+)\_(\d+)\.*'
            m = re.match(pattern, path.stem)
            if m is None: continue
            grammar_type, clustering, mu = m.groups()

            print(grammar_type, clustering, mu, end='\t', flush=True)

            if clustering in root_dict[name]:
                ht, avg_branch_factor, dc = root_dict[name][clustering]
            else:
                root = load_pickle(f'{base_path}/output/trees/{name}/{clustering}_list.pkl')
                if root is None:
                    continue
                if isinstance(root, list): root = create_tree(root)
                ht, avg_branch_factor, dc = get_tree_stats(g=orig_graph, root=root)
                root_dict[name][clustering] = ht, avg_branch_factor, dc

            vrg = load_pickle(fname)
            if vrg is None: continue
            graph_dl = dl_dict[f'{name}-plain'] if grammar_type.startswith('VRG') else dl_dict[name]
            if 'all-tnodes' in grammar_type:
                mu = -1
            row = dict(name=name, orig_n=orig_graph.order(), orig_m=orig_graph.size(), attr_name=attr_name,
                       model=grammar_type, mu=int(mu), clustering=clustering, cost=dc, branch_factor=avg_branch_factor,
                       height=ht, graph_dl=graph_dl, num_rules=vrg.num_rules, unique_rules=len(vrg.unique_rule_list),
                       grammar_dl=vrg.cost)
            rows.append(row)
        temp_df = pd.DataFrame(rows)
        temp_df.to_csv(temp_fname, index=False)

    # for name in names:
    #     temp_fname = f'{base_path}/stats/_grammar_df_{name}.csv'
    #     if Path(temp_fname).exists():
    #         os.remove(temp_fname)
    return pd.DataFrame(rows)


def make_combined_graph_dfs(outdir, names, clusterings, bipartite=False):
    dfs = []
    mus = range(3, 11)
    # mus = range(5, 8)

    possbile_models = 'SBM', 'DC-SBM', 'CL', 'AGM', 'NetGAN', 'CELL'
    for name in names:  # add tqdm
        temp_fname = f'{base_path}/stats/_graph_df_{name}.csv'
        if Path(temp_fname).exists():
            print(f'Skipping {name!r}')
            continue

        sub_df = []
        orig_graph, attr_name = get_graph(name, basedir=outdir)

        for fname in glob.glob(f'{outdir}/output/graphs/{name}/*'):
            path = Path(fname)

            if path.stem.startswith(possbile_models) or 'ae' in path.stem:
                print(path.stem, end='\t', flush=True)
                pattern = r'(.*)\_(\d+)'
                m = re.match(pattern, path.stem)
                if m is None: continue
                grammar_type, _ = m.groups()
                mu = np.nan
                clustering = np.nan

            elif path.stem.startswith('VRG') or path.stem.startswith('AVRG'):
                pattern = r'(.*)\_(\w+)\_(\d+)\_(\d+)'
                m = re.match(pattern, path.stem)
                if m is None: continue
                grammar_type, clustering, mu, _ = m.groups()
                mu = int(mu)
                if mu not in mus or clustering not in clusterings: continue
                print(path.stem, end='\t', flush=True)
            else:
                assert NotImplementedError(f'Invalid model: {path.stem!r}')
                continue

            df = make_graph_df(name, fname, orig_graph, mu, clustering, attr_name, grammar_type, bipartite)
            dfs.append(df)
            sub_df.append(df)
            # df.to_csv(temp_fname, index=False)
        if len(sub_df) > 0:
            temp_df = pd.concat(sub_df, ignore_index=True)
            temp_df.to_csv(temp_fname)
            print(f'Writing {name!r} to {temp_fname!r}')

    # os.remove(temp_df_filename)
    graph_df = pd.concat(dfs, ignore_index=True)
    return graph_df


def filter_terminal_graph(graph):
    terminal_nodes = [n for n, d in graph.nodes(data=True) if 'nt' not in d]
    return graph.subgraph(terminal_nodes).copy()


def un_nest_attr_dict(g):
    new_g = nx.Graph()
    for n, d in g.nodes(data=True):
        while 'attr_dict' in d:
            d = d['attr_dict']
        new_g.add_node(n, **d)
    new_g.add_edges_from(g.edges())
    return new_g


def make_gen_df(base_path, names=None, clusterings=None, num_samples=None):
    """
    num_samples is for number of samples of generated graph
    """
    rows = []
    cols = ['snap_id', 'name', 'model', 'clustering', 'attr_name',
            'orig_n', 'orig_m', 'orig_deg_ast', 'orig_attr_ast',
            'mu', 'n', 'm', 't', 'term_graph',
            'term_n', 'term_m', 'term_degree_js', 'term_pagerank_js', 'term_lambda_dist',
            'term_deg_ast', 'term_attr_ast']

    if names is None:
        names = ['karate', 'football', 'polbooks', 'us-flights', 'cora', 'citeseer','polblogs', 'pubmed'][: -1]

    # mus = [5, 6]
    mus = range(3, 11)
    snap_id = 0  # snap id is the track of generated graphs - 10

    if clusterings is None:
        clusterings = ['cond', 'leiden', 'spectral', 'consensus']

    for name in names:
        orig_graph, attr_name = get_graph(name, basedir=base_path)
        orig_deg_ast = nx.degree_assortativity_coefficient(orig_graph)
        orig_att_ast = nx.attribute_assortativity_coefficient(orig_graph, attr_name)

        orig_gstats = GraphStats(orig_graph)

        for gen_filename in glob.glob(f'{base_path}/output/generators/{name}/*'):
            path = Path(gen_filename)
            gen: RandomGenerator = load_pickle(path)  # all gen snapshots has 10 different generations - we need maybe just 1
            if gen is None: continue

            print(path.stem, end='\t', flush=True)
            pattern = r'(.*)\_(\w+)\_(\d+)\_(\d+)'
            m = re.match(pattern, path.stem)
            grammar_type, clustering, mu, _ = m.groups()
            mu = int(mu)
            if mu not in mus or clustering not in clusterings: continue

            generated_graph_snapshots = gen.all_gen_snapshots[0]
            del gen  # delete the object to save memory

            if num_samples is None: num_samples = len(generated_graph_snapshots)

            indices = sorted(set(np.linspace(0, len(generated_graph_snapshots) - 1,
                                             num_samples, dtype=int, endpoint=True)))

            for t in indices:
                graph = generated_graph_snapshots[t]
                terminal_graph = filter_terminal_graph(graph)
                terminal_graph = un_nest_attr_dict(terminal_graph)
                row = {col: np.nan for col in cols}

                row.update(dict(snap_id=snap_id, name=name,
                                model=grammar_type, clustering=clustering, attr_name=attr_name,
                                orig_n=orig_graph.order(), orig_m=orig_graph.size(),
                                orig_deg_ast=orig_deg_ast, orig_attr_ast=orig_att_ast,
                                mu=mu, t=t, n=graph.order(), m=graph.size(), term_graph=terminal_graph,
                                term_n=terminal_graph.order(), term_m=terminal_graph.size()))

                if terminal_graph.size() > 0:
                    gen_gstats = GraphStats(terminal_graph)

                    gpc = GraphPairCompare(orig_gstats, gen_gstats)
                    row.update(term_degree_js=gpc.degree_js(), term_pagerank_js=gpc.pagerank_js(),
                               term_lambda_dist=gpc.lambda_dist(),
                               term_deg_ast=nx.degree_assortativity_coefficient(terminal_graph),
                               term_attr_ast=nx.attribute_assortativity_coefficient(terminal_graph, attr_name),
                               deg_mix_dist_dict=gpc.deg_mixing_dist_dict(),
                               attr_mix_dist_dict=gpc.attr_mixing_dist_dict())
                rows.append(row)
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(f'{base_path}/stats/temp_gen_df.csv', index=False)
    return pd.DataFrame(rows)


def get_matchine_name(home):
    with open(os.path.join(home, '.name'), 'r') as fp:
        lines = fp.readlines()[0].strip()
    return lines


if __name__ == '__main__':
    names = ['karate', 'football', 'polbooks', 'wisconsin', 'texas', 'cornell', 'airports',
             'polblogs', 'cora', 'citeseer', 'chameleon', 'film', 'squirrel', 'pubmed'][1: ]

    clusterings = ['cond', 'leiden', 'louvain', 'spectral', 'infomap', 'labelprop', 'random',
                   'leadingeig', 'consensus'][: 3]

    home = Path.home()
    machine = get_matchine_name(home)

    if machine == 'dsg3':
        base_path = '/data/ssikdar/Attributed-VRG'
    elif 'dan' in machine:
        base_path = '/data/ssikdar/Attributed-VRG'
    else:
        raise BaseException(f'No machine name')

    # grammar_path = f'{base_path}/stats/grammar_df.csv'
    # if not check_file_exists(grammar_path):
    #     grammar_df = make_grammar_df(base_path=base_path, names=names)
    #     grammar_df.to_csv(grammar_path, index=False)

    graphs_path = f'{base_path}/stats/graphs_df_all_models.csv'
    if not check_file_exists(graphs_path):
        graphs_df = make_combined_graph_dfs(outdir=base_path, names=names, clusterings=clusterings, bipartite=False)
        graphs_df.to_csv(graphs_path, index=False)

    exit(0)
    # gen_df_path = f'{base_path}/stats/gen_df.csv'
    # num_samples = 10
    # if not check_file_exists(gen_df_path):
    #     gen_df = make_gen_df(base_path=base_path, names=names, clusterings=clusterings, num_samples=num_samples)
    #     gen_df.to_csv(gen_df_path, index=False)

    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'random', 'infomap', 'labelprop',
    #                'leadingeig', 'consensus'][: 5]
    #
