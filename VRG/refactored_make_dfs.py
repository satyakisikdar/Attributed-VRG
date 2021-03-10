import glob
import logging
import re
from os.path import join
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns;
from tqdm import tqdm

from VRG.src.HierClustering import HierarchicalClustering
from VRG.src.VRG import AttributedVRG

sns.set_style('white')
import sys;

sys.path.extend(['../', '../../', ])
from VRG.src.generate import RandomGenerator

from anytree import LevelOrderIter
from statistics import mean
import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6

from VRG.src.utils import load_pickle, dump_pickle
from VRG.src.graph_stats import GraphStats
from VRG.src.graph_comparison import GraphPairCompare
from VRG.refactored_runner import read_graph
from VRG.src.Tree import dasgupta_cost
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
    H = np.nan
    # Y = np.zeros(shape=(g.order(), len(mapping)))
    # for n, d in g.nodes(data=True):
    #     attr = d[attr_name]
    #     Y[n, mapping[attr]] = 1
    # A = nx.adjacency_matrix(g)
    # E = np.ones(shape=(A.shape[0], len(mapping)))
    #
    # H = (Y.T @ A @ Y) / (Y.T @ A @ E)

    return_d = dict(homophily_ratio=h, compatibility_mat=H, attr_name=attr_name, mapping=mapping)
    return return_d


def get_tree_stats(g, root, cost=False):
    """
    Compute height of the tree, avg branching factor, dasgupta cost
    """
    ht = root.height
    avg_branch_factor = mean(len(node.children) for node in LevelOrderIter(root) if len(node.children) > 1)
    if cost:
        dc = dasgupta_cost(g, root)
    else:
        dc = None
    return ht, avg_branch_factor, dc


def _get_basic_stats(gstats: GraphStats, kind: str) -> dict:
    row = dict(n=gstats.graph.order(), m=gstats.graph.size(),
               deg_ast=gstats['degree_assortativity'],
               attr_ast=gstats['attribute_assortativity'],
               deg_mix_dict=gstats['degree_mixing_dict'], attr_mix_dict=gstats['attr_mixing_dict'])
    formatted_row = {f'{kind}_{key}': val for key, val in row.items()}
    return formatted_row


def refactored_grammar_df(basedir: str, names: List[str], clusterings: List[str], extract_types: List[str],
                          mus: List[int], overwrite: bool):
    base_grammars_path = Path(basedir) / 'output/grammars'

    input_graph_dls = {name: graph_mdl(read_graph(name=name, basedir=basedir), attributed=True) for name in names}
    dasgupta_costs = {name: {} for name in names}

    for name in names:
        grammars_path = base_grammars_path / name
        stats_path = Path(basedir) / 'stats' / 'grammars' / f'_temp_{name}_.csv'

        if not overwrite and stats_path.exists():
            logging.error(f'Existing CSV found {stats_path!r}, skipping')
            continue

        rows = []
        for grammar_filename in grammars_path.glob('*.pkl'):
            avrg: AttributedVRG = load_pickle(grammar_filename)
            if avrg.clustering not in clusterings or avrg.extract_type not in extract_types or avrg.mu not in mus:
                continue

            grammar_args = avrg.grammar_args
            input_graph = grammar_args.input_graph
            hc_obj: HierarchicalClustering = grammar_args.hc_obj

            if hc_obj.stats['cost'] == -1 and avrg.clustering not in dasgupta_costs[name]:
                hc_obj.get_clustering()  # the root gets reset
                hc_obj.calculate_cost()

            dasgupta_costs[name][avrg.clustering] = hc_obj.stats['cost']
            tree_stats = grammar_args.hc_obj.stats
            row = dict(name=name, orig_n=input_graph.order(), orig_m=input_graph.size(),
                       input_graph_dl=input_graph_dls[name],
                       extract_type=avrg.extract_type, mu=avrg.mu, clustering=avrg.clustering,
                       height=tree_stats['height'], avg_branch_fac=tree_stats['avg_branch_factor'],
                       median_branch_fac=tree_stats['median_branch_factor'],
                       dasgupta_cost=dasgupta_costs[name][avrg.clustering],
                       num_rules=avrg.num_rules, grammar_dl=avrg.cost)
            rows.append(row)
        if len(rows) > 0:
            logging.error(f'Writing Grammar DF for {name!r} at {stats_path!r}')
            df = pd.DataFrame(rows)
            df.to_csv(stats_path, index=False)
    return


def make_graph_df_new(name: str, fname: str, basedir: str, orig_gstats: GraphStats, slow_stats: bool):
    attr_name = 'value'
    gen_graphs = load_pickle(fname)
    if gen_graphs is None:
        return pd.DataFrame()

    # break down the filename to figure out the different parts
    path = Path(fname)
    pattern = r'(.+)_(.+)_(.+)_(.+)_(\d+).*'
    m = re.match(pattern, path.stem)
    if m is None:
        return
    gen_type, extract_type, clustering, mu, _ = m.groups()

    # if fast is enabled, only calculate measures that are easy to compute 'fast' methods
    # degree_js, pagerank_js, lambda_dist, degree_ast, deg_mixing_dict, attr_ast, attr_mix_dict
    fast_stats = ['degree_js', 'pagerank_js', 'lambda_dist', 'deg_ast', 'deg_mix_dict', 'attr_ast', 'attr_mix_dict']

    if slow_stats:
        orig_sp_ast_5 = orig_gstats.shortest_path_ast(alpha=0.05, fname=join(basedir, 'input', f'{name}.gml'))
        orig_sp_ast_50 = orig_gstats.shortest_path_ast(alpha=0.5, fname=join(basedir, 'input', f'{name}.gml'))
        orig_sp_ast_100 = orig_gstats.shortest_path_ast(alpha=1, fname=join(basedir, 'input', f'{name}.gml'))
        orig_sp_ast_500 = orig_gstats.shortest_path_ast(alpha=5, fname=join(basedir, 'input', f'{name}.gml'))
        orig_sp_ast_1000 = orig_gstats.shortest_path_ast(alpha=10, fname=join(basedir, 'input', f'{name}.gml'))

    rows = []
    orig_stats = _get_basic_stats(gstats=orig_gstats, kind='orig')  # add basic stats of the original graph
    for g in gen_graphs:
        orig_graph = orig_gstats.graph
        row = dict(name=name, orig_graph=orig_graph, gen_type=gen_type, extract_type=extract_type,
                   clustering=clustering, mu=mu)
        row.update(orig_stats)  # add the original stats

        gen_gstats = GraphStats(g)
        row.update(_get_basic_stats(gen_gstats, kind='gen'))

        gpc = GraphPairCompare(orig_gstats, gen_gstats)
        row.update(dict(degree_js=gpc.degree_js(), pagerank_js=gpc.pagerank_js(), lambda_dist=gpc.lambda_dist()))

        if slow_stats:
            gen_sp_ast_5 = gen_gstats.shortest_path_ast(alpha=0.05)
            gen_sp_ast_50 = gen_gstats.shortest_path_ast(alpha=0.5)
            gen_sp_ast_100 = gen_gstats.shortest_path_ast(alpha=1)
            gen_sp_ast_500 = gen_gstats.shortest_path_ast(alpha=5)
            gen_sp_ast_1000 = gen_gstats.shortest_path_ast(alpha=10)
            row.update(dict(orig_sp_ast_5=orig_sp_ast_5, orig_sp_ast_50=orig_sp_ast_50,
                            orig_sp_ast_100=orig_sp_ast_100, orig_sp_ast_500=orig_sp_ast_500,
                            orig_sp_ast_1000=orig_sp_ast_1000))

            row.update(dict(gen_sp_ast_5=gen_sp_ast_5, gen_sp_ast_50=gen_sp_ast_50, gen_sp_ast_100=gen_sp_ast_100,
                            gen_sp_ast_500=gen_sp_ast_500, gen_sp_ast_1000=gen_sp_ast_1000))

        rows.append(row)
    return pd.DataFrame(rows)


def make_grammar_df(basedir, names, clusterings, overwrite):
    root_dict_pickle_fname = join(basedir, 'input', 'root_dict.pkl')
    root_dict = load_pickle(root_dict_pickle_fname)
    if root_dict is None:
        root_dict = {}
        recompute = True
    else:
        recompute = False
        for name in names:
            if name not in root_dict:
                recompute = True
            for clustering in clusterings:
                if clustering not in root_dict[name]:
                    recompute = True

    if recompute:
        for name in tqdm(names, desc='Name'):
            orig_graph = read_graph(name, basedir=basedir)
            if name not in root_dict:
                root_dict[name] = {}
            for clustering in tqdm(clusterings, desc='Clustering', leave=False):
                if clustering in root_dict[name]:
                    continue
                root = load_pickle(f'{basedir}/output/trees/{name}/{clustering}_root.pkl')
                if root is None:
                    continue
                ht, avg_branch_factor, dc = get_tree_stats(g=orig_graph, root=root, cost=True)
                print(dc)
                # dc = cost_dict[name][clustering]
                root_dict[name][clustering] = ht, avg_branch_factor, dc
            dump_pickle(root_dict, root_dict_pickle_fname)

    print(root_dict)
    dl_dict = {}

    for name in names:
        temp_fname = f'{basedir}/stats/temp/_grammar_df_{name}.csv'
        if Path(temp_fname).exists() and not overwrite:
            print(f'Skipping {name!r}')
            continue

        orig_graph = read_graph(name, basedir=basedir)
        dl_dict[name] = graph_mdl(orig_graph, attributed=True)
        rows = []

        print('\n\n', name)
        files = glob.glob(f'{basedir}/output/grammars/{name}/*.pkl')
        for fname in tqdm(files, total=len(files), desc=f'{name}'):
            path = Path(fname)
            pattern = r'(\w+)_(.+)\_(\w+)_(.+).*'
            m = re.match(pattern, path.stem)
            if m is None:
                continue
            grammar_type, extract_type, clustering, mu = m.groups()
            if clustering not in clusterings:  # skip over clusterings we dont care about
                continue

            if grammar_type.startswith('VRG'):  # skip over regular VRGs
                continue

            tqdm.write(f'{grammar_type}, {extract_type}, {clustering}, {mu}')
            ht, avg_branch_factor, dc = root_dict[name][clustering]

            vrg = load_pickle(fname)
            if vrg is None:
                continue
            graph_dl = dl_dict[name]

            row = dict(name=name, orig_n=orig_graph.order(), orig_m=orig_graph.size(), grammar_type=grammar_type,
                       extract_type=vrg.extract_type, mu=int(mu), clustering=clustering, cost=dc,
                       branch_factor=avg_branch_factor,
                       height=ht, graph_dl=graph_dl, num_rules=vrg.num_rules,
                       grammar_dl=vrg.cost)
            rows.append(row)
        if len(rows) > 0:
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(temp_fname, index=False)

    # for name in names:
    #     temp_fname = f'{base_path}/stats/_grammar_df_{name}.csv'
    #     if Path(temp_fname).exists():
    #         os.remove(temp_fname)
    return pd.DataFrame(rows)


def make_cabam_grammar_df(basedir, graphs, clusterings, overwrite):
    names = [f'cabam-{i}' for i in range(len(graphs))]
    dl_dict = {name: {} for name in names}

    for name, orig_graph in zip(names, graphs):
        temp_fname = f'{basedir}/stats/temp/cabam/_grammar_df_{name}.csv'
        if Path(temp_fname).exists() and not overwrite:
            print(f'Skipping {name!r}')
            continue

        dl_dict[name] = graph_mdl(orig_graph, attributed=True)
        rows = []

        print('\n\n', name)
        files = glob.glob(f'{basedir}/output/grammars/cabam/{name}_*.pkl')
        for fname in tqdm(files, total=len(files), desc=f'{name}'):
            path = Path(fname)
            pattern = r'(.+)_(\w+)_(.+)\_(\w+)_(.+).*'
            m = re.match(pattern, path.stem)
            if m is None:
                continue
            name, grammar_type, extract_type, clustering, mu = m.groups()
            if clustering not in clusterings:  # skip over clusterings we dont care about
                continue

            if grammar_type.startswith('VRG'):  # skip over regular VRGs
                continue

            tqdm.write(f'{grammar_type}, {extract_type}, {clustering}, {mu}')
            vrg = load_pickle(fname)
            if vrg is None:
                continue

            root = load_pickle(join(basedir, 'output/trees/cabam', f'{name}_{clustering}_root.pkl'))
            ht, avg_branch_factor, dc = get_tree_stats(g=orig_graph, root=root, cost=True)
            graph_dl = dl_dict[name]

            row = dict(name=name, orig_n=orig_graph.order(), orig_m=orig_graph.size(), grammar_type=grammar_type,
                       extract_type=vrg.extract_type, mu=int(mu), clustering=clustering, cost=dc,
                       branch_factor=avg_branch_factor, height=ht, graph_dl=graph_dl, num_rules=vrg.num_rules,
                       grammar_dl=vrg.cost)
            rows.append(row)
        if len(rows) > 0:
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(temp_fname, index=False)

    # for name in names:
    #     temp_fname = f'{base_path}/stats/_grammar_df_{name}.csv'
    #     if Path(temp_fname).exists():
    #         os.remove(temp_fname)
    return pd.DataFrame(rows)


def make_cabam_graph_dfs(basedir, graphs, clusterings, final=False, slow_stats=False, mus=None, extract_types=None):
    dfs = []

    if mus is None:
        mus = list(range(3, 11)) + [0]

    if extract_types is None:
        extract_types = ['mu-random', 'mu-level', 'all-tnodes']

    write_every = 10
    names = [f'cabam-{idx}' for idx in range(len(graphs))]

    for name, orig_graph in zip(names, graphs):
        if final:
            temp_fname = f'{basedir}/stats/temp/graphs/_final_graph_df_{name}.csv'
        else:
            temp_fname = f'{basedir}/stats/temp/graphs/_graph_df_{name}.csv'

        existing_df = pd.read_csv(temp_fname) if Path(temp_fname).exists() else None
        orig_gstats = GraphStats(orig_graph)
        i = 0

        for fname in tqdm(glob.glob(f'{basedir}/output/graphs/cabam/{name}_*'), desc=f'{name}', ncols=100):
            path = Path(fname)
            if not path.stem.startswith('AVRG'):
                continue

            pattern = r'(.+)_(.+)_(.+)_(.+)_(.+)_(.+)_(.*).pkl'
            m = re.match(pattern, path.stem)
            if m is None:
                return
            name, _, extract_type, clustering, mu, gen_type, extract_type, clustering, mu, _ = m.groups()
            mu = int(mu)

            if clustering not in clusterings or mu not in mus or extract_type not in extract_types:
                continue

            ## check if the row exists already
            if existing_df is not None:
                if not existing_df[(existing_df.name == name) &
                                   (existing_df.gen_type == gen_type) &
                                   (existing_df.extract_type == extract_type) &
                                   (existing_df.clustering == clustering) &
                                   (existing_df.mu == mu)].empty:
                    continue

            tqdm.write(path.stem)

            df = make_graph_df_new(basedir=basedir, fname=fname, name=name, orig_gstats=orig_gstats,
                                   slow_stats=slow_stats)
            # add df to the existing df
            if existing_df is None:
                existing_df = df
            else:
                existing_df = existing_df.append(df, ignore_index=True)

            if (i > 0) and (i % write_every == 0):
                tqdm.write(f'Writing partial results {name}!')
                existing_df.to_csv(temp_fname, index=False)
            i += 1

        # write existing df again
        existing_df.to_csv(temp_fname, index=False)
        print(f'Writing {name!r} to {temp_fname!r}')

    # os.remove(temp_df_filename)
    if len(dfs) > 0:
        graph_df = pd.concat(dfs, ignore_index=True)
    else:
        graph_df = None
    return graph_df


def make_all_graph_dfs(basedir, names, clusterings, final=False, slow_stats=False, mus=None, extract_types=None):
    dfs = []

    if mus is None:
        mus = list(range(3, 11)) + [0]

    if extract_types is None:
        extract_types = ['mu-random', 'mu-level', 'all-tnodes']

    write_every = 10

    for name in names:
        if final:
            temp_fname = f'{basedir}/stats/temp/graphs/_final_graph_df_{name}.csv'
        else:
            temp_fname = f'{basedir}/stats/temp/graphs/_graph_df_{name}.csv'

        existing_df = pd.read_csv(temp_fname) if Path(temp_fname).exists() else None

        orig_graph = read_graph(name, basedir=basedir)
        orig_gstats = GraphStats(orig_graph)
        i = 0

        for fname in tqdm(glob.glob(f'{basedir}/output/graphs/{name}/*'), desc=f'{name}', ncols=100):
            path = Path(fname)
            if not path.stem.startswith('AVRG'):
                continue

            pattern = r'(.+)_(.+)_(.+)_(.+)_(\d+).*'
            m = re.match(pattern, path.stem)
            if m is None:
                return
            gen_type, extract_type, clustering, mu, _ = m.groups()
            mu = int(mu)

            if clustering not in clusterings or mu not in mus or extract_type not in extract_types:
                continue

            ## check if the row exists already
            if existing_df is not None:
                if not existing_df[(existing_df.name == name) &
                                   (existing_df.gen_type == gen_type) &
                                   (existing_df.extract_type == extract_type) &
                                   (existing_df.clustering == clustering) &
                                   (existing_df.mu == mu)].empty:
                    continue

            tqdm.write(path.stem)

            df = make_graph_df_new(basedir=basedir, fname=fname, name=name, orig_gstats=orig_gstats,
                                   slow_stats=slow_stats)
            # add df to the existing df
            if existing_df is None:
                existing_df = df
            else:
                existing_df = existing_df.append(df, ignore_index=True)

            if (i > 0) and (i % write_every == 0):
                tqdm.write(f'Writing partial results {name}!')
                existing_df.to_csv(temp_fname, index=False)
            i += 1

        # write existing df again
        existing_df.to_csv(temp_fname, index=False)
        print(f'Writing {name!r} to {temp_fname!r}')

    # os.remove(temp_df_filename)
    if len(dfs) > 0:
        graph_df = pd.concat(dfs, ignore_index=True)
    else:
        graph_df = None
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
        names = ['karate', 'football', 'polbooks', 'us-flights', 'cora', 'citeseer', 'polblogs', 'pubmed'][: -1]

    # mus = [5, 6]
    mus = range(3, 11)
    snap_id = 0  # snap id is the track of generated graphs - 10

    if clusterings is None:
        clusterings = ['cond', 'leiden', 'spectral', 'consensus']

    for name in names:
        orig_graph = read_graph(name, basedir=base_path)
        attr_name = 'value'
        orig_deg_ast = nx.degree_assortativity_coefficient(orig_graph)
        orig_att_ast = nx.attribute_assortativity_coefficient(orig_graph, attr_name)

        orig_gstats = GraphStats(orig_graph)

        for gen_filename in glob.glob(f'{base_path}/output/generators/{name}/*'):
            path = Path(gen_filename)
            gen: RandomGenerator = load_pickle(
                path)  # all gen snapshots has 10 different generations - we need maybe just 1
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


def get_final_graph_dfs():
    basedir = '/data/ssikdar/Attributed-VRG'

    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports',
             'polblogs', 'film', 'chameleon', 'squirrel']
    clusterings = ['leiden']
    extract_types = ['mu-random']
    mus = [5]

    graphs_df = make_all_graph_dfs_new(basedir=basedir, names=names, clusterings=clusterings, mus=mus,
                                       extract_types=extract_types, final=True, slow_stats=True)
    return


def refactored_grammar_df_runner():
    basedir = '/data/ssikdar/AVRG'
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports',
             'polblogs', 'film', 'chameleon', 'squirrel']

    clusterings = ['cond', 'spectral', 'hyphc', 'leiden', 'louvain', 'infomap', 'labelprop', 'random']

    # clusterings = ['cond', 'leiden', 'hyphc']
    # mus = [0, 5]
    mus = list(range(3, 11)) + [0]
    extract_types = ['mu-level', 'mu-random', 'all-tnodes']
    overwrite = False
    refactored_grammar_df(basedir=basedir, names=names, clusterings=clusterings, extract_types=extract_types, mus=mus,
                          overwrite=overwrite)
    return


def old_main():
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports',
             'polblogs', 'film', 'chameleon', 'squirrel']
    # names = ['citeseer']

    clusterings = ['cond', 'leiden', 'louvain', 'spectral', 'infomap', 'labelprop', 'random']  # [: 3]

    basedir = '/data/ssikdar/Attributed-VRG'
    # grammar_path = f'{base_path}/stats/grammar_df.csv'
    # if not check_file_exists(grammar_path):
    # grammar_df = make_grammar_df(base_path=basedir, names=names, clusterings=clusterings, overwrite=True)
    # grammar_df.to_csv(grammar_path, index=False)

    # graphs_path = f'{base_path}/stats/graphs_df_all_models.csv'
    # if not check_file_exists(graphs_path):
    clusterings = ['cond', 'leiden', 'louvain']
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', ]
    graphs_df = make_all_graph_dfs_new(basedir=basedir, names=names, clusterings=clusterings, overwrite=False)
    #     graphs_df.to_csv(graphs_path, index=False)

    exit(0)
    # gen_df_path = f'{base_path}/stats/gen_df.csv'
    # num_samples = 10
    # if not check_file_exists(gen_df_path):
    #     gen_df = make_gen_df(base_path=base_path, names=names, clusterings=clusterings, num_samples=num_samples)
    #     gen_df.to_csv(gen_df_path, index=False)

    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'random', 'infomap', 'labelprop',
    #                'leadingeig', 'consensus'][: 5]
    #


def cabam():
    basedir = '/data/ssikdar/AVRG'
    cabam_graphs = load_pickle(join(basedir, 'input', 'cabam.graphs'))
    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random']
    clusterings = ['cond', 'leiden', 'hyphc']
    mus = [0, 5]
    # make_cabam_grammar_df(basedir=basedir, graphs=cabam_graphs, clusterings=clusterings, overwrite=False)
    make_cabam_graph_dfs(basedir=basedir, graphs=cabam_graphs, clusterings=clusterings, mus=mus)
    return


if __name__ == '__main__':
    # get_final_graph_dfs()
    # cabam()
    refactored_grammar_df_runner()
