import argparse
import logging
import math
import os
import sys
from os.path import join
from pathlib import Path
from time import time
from typing import Any, List, Union, Dict, Tuple

import networkx as nx
import seaborn as sns
from tqdm import tqdm

sys.path.extend(['../', '../../', './', '../../../', '/home/ssikdar/tmp_dir'])

import VRG.src.partitions as partitions
from VRG.src.LightMultiGraph import LightMultiGraph
from VRG.src.MDL import graph_dl
from VRG.src.Tree import create_tree
from VRG.src.VRG import VRG, NCE, AttributedVRG
from VRG.src.consensus_clustering import get_consensus_root
from VRG.src.extract import NCEExtractor, VRGExtractor, AVRGExtractor
from VRG.src.generate import RandomGenerator, NCEGenerator, AttributedRandomGenerator, GreedyAttributeRandomGenerator
from VRG.src.utils import dump_pickle, check_file_exists, load_pickle, timer, nx_to_lmg, get_mixing_dict, CustomCounter

sys.setrecursionlimit(1_000_000)
# logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True
sns.set_style('white')


def get_graph(gname: str, basedir: str) -> Tuple[nx.Graph, str]:
    start_time = time()
    attr_name = ''
    if gname == 'sample':
        g = nx.Graph()
        g.add_nodes_from(range(5), value='blue')
        g.add_nodes_from(range(5, 9), value='red')

        g.add_edges_from([(0, 1), (0, 3), (0, 4),
                          (1, 2), (1, 4), (1, 5),
                          (2, 3), (2, 4), (2, 8),
                          (3, 4),
                          (5, 6), (5, 7), (5, 8),
                          (6, 7), (6, 8),
                          (7, 8)])  # properly labeled
        g.name = 'sample'
        attr_name = 'value'
    elif gname.endswith('.gpickle'):
        g = nx.read_gpickle(gname)
        g.name = Path(gname).stem
    else:
        if gname in ('waterloo', 'grenoble', 'uppsala'):
            g = nx.read_gpickle(f'../snap_data/cleaned/{gname}_lcc_attr.gpickle')
        try:
            g = nx.read_gml(join(basedir, 'input', f'{gname}.gml'))
            attr_name = 'value'
        except Exception:
            path = join(basedir, 'input', f'{gname}.g')
            g = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())

        g.remove_edges_from(nx.selfloop_edges(g))
        if not nx.is_connected(g):
            nodes_lcc = max(nx.connected_components(g), key=len)
            g = g.subgraph(nodes_lcc).copy()
        name = g.name
        g = nx.convert_node_labels_to_integers(g, label_attribute='orig_label')
        g.name = name

    end_time = round(time() - start_time, 2)
    dl = graph_dl(g)
    logging.error(f'{gname!r}, n = {g.order():,d}, m = {g.size():,d}, dl = {dl:,g} bits read in {round(end_time, 3):,g}s.')

    return g, attr_name


@timer
def get_clustering(g: nx.Graph, outdir: str, clustering: str, use_pickle: bool, filename='') -> Any:
    """
    wrapper method for getting dendrogram. uses an existing pickle if it can.
    :param g: graph
    :param outdir: output directory where picles are stored
    :param clustering: name of clustering method
    :param use_pickle: flag to whether or not to use the pickle
    :return: root node of the dendrogram
    """
    if g.name == 'sample':
        list_of_list_clusters = [
            [
                [[0], [1]],
                [[2], [[3], [4]]]
            ],
            [
                [[5], [6]],
                [[7], [8]]
            ]
        ]
        return list_of_list_clusters

    if filename == '':
        list_of_list_filename = os.path.join(outdir, 'output', 'trees', g.name, f'{clustering}_list.pkl')
    else:
        list_of_list_filename = filename

    if check_file_exists(list_of_list_filename) and use_pickle:
        logging.error(f'Using existing pickle for {clustering!r} clustering\n')
        list_of_list_clusters = load_pickle(list_of_list_filename)

    else:
        tqdm.write(f'Running {clustering!r} clustering on {g.name!r}...')
        if clustering == 'random':
            list_of_list_clusters = partitions.get_random_partition(g)
        elif clustering == 'consensus':
            # delete the matlab tree and sc files
            matlab_files_path = './src/matlab_clustering/HierarchicalConsensus/data'
            tree_path = os.path.join(matlab_files_path, f'{g.name}_tree.mat')
            sc_path = os.path.join(matlab_files_path, f'{g.name}_sc.vec')
            if check_file_exists(tree_path):
                os.remove(tree_path)
            if check_file_exists(sc_path):
                os.remove(sc_path)
            list_of_list_clusters = get_consensus_root(g=g, gname=g.name)
        elif clustering in ('leiden', 'louvain', 'infomap', 'labelprop', 'leadingeig'):
            try:
                list_of_list_clusters = partitions.louvain_leiden_infomap_label_prop(g, method=clustering)
            except Exception as e:
                list_of_list_clusters = []
        elif clustering == 'cond':
            list_of_list_clusters = partitions.approx_min_conductance_partitioning(g)
        elif clustering == 'spectral':
            list_of_list_clusters = partitions.spectral_kmeans(g, K=int(math.sqrt(g.order() // 2)))
        else:
            raise NotImplementedError(f'Invalid clustering algorithm {clustering!r}')

        if len(list_of_list_clusters) != 0: dump_pickle(list_of_list_clusters, list_of_list_filename)

    return list_of_list_clusters


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


def get_grammars(name: str, clustering: str, grammar_type: Tuple[str, str], mu: int, input_graph: nx.Graph,
                 use_grammar_pickle: bool, use_cluster_pickle: bool, attr_name: str, outdir: str = 'dumps', count: int = 1,
                 grammar_filename: str = '', write_pickle: bool = True, list_of_list_clusters=None) -> List[Union[VRG, NCE]]:
    """
    Dump the stats
    :return:
    """
    if input_graph.name != name:
        input_graph.name = name
    # make_dirs(outdir, name)  # make the directories if needed

    # print(f'Extracting {count} grammars')
    grammars = []

    for i in range(count):
        if grammar_filename == '':
            grammar_filename = f'{outdir}/output/grammars/{name}/{grammar_type[0]}-{grammar_type[1].replace("_", "-")}_{clustering}_{mu}_{i}.pkl'
        logging.error(f'Extracting grammar: {grammar_filename}')
        if use_grammar_pickle and check_file_exists(grammar_filename):
            logging.error(f'Using pickled grammar from {grammar_filename!r}')
            grammar = load_pickle(grammar_filename)
        else:
            if list_of_list_clusters is None:
                list_of_list_filename = os.path.join(outdir, 'output', 'trees', input_graph.name, f'{clustering}_list.pkl')
                if not Path(list_of_list_filename).exists():
                    logging.error(f'Skipping grammar, name {input_graph.name!r} clustering {clustering!r}')
                    continue
                list_of_list_clusters = get_clustering(g=input_graph, outdir=outdir,
                                                       clustering=clustering, use_pickle=use_cluster_pickle,
                                                       filename=list_of_list_filename)
            root = create_tree(list_of_list_clusters) if isinstance(list_of_list_clusters, list) else list_of_list_clusters
            # dc = dasgupta_cost(g=g, root=root, use_parallel=True)
            lmg: LightMultiGraph = nx_to_lmg(nx_g=input_graph)

            if grammar_type[0] == 'VRG':
                extractor = VRGExtractor(g=lmg, type=grammar_type[1], mu=mu, root=root, clustering=clustering)
            elif grammar_type[0] == 'NCE':
                extractor = NCEExtractor(g=lmg, type=grammar_type[1], mu=mu, root=root, clustering=clustering)
            elif grammar_type[0] == 'AVRG':
                assert attr_name != ''
                extractor = AVRGExtractor(g=lmg, attr_name=attr_name, type=grammar_type[1], clustering=clustering,
                                          mu=mu, root=root)
            else:
                raise NotImplementedError(f'Invalid grammar type {grammar_type!r}')

            grammar = extractor.extract()
            logging.error(str(grammar))
            if write_pickle: dump_pickle(grammar, grammar_filename)
        grammars.append(grammar)
    return grammars


def generate_graphs(name: str, grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, grammar_type: str, outdir: str = 'dumps',
                    mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, fancy=None,
                    inp_deg_ast: float = None, inp_attr_ast: float = None, use_pickle: bool = False,
                    save_snapshots: bool = False, alpha: Union[None, float] = None, graphs_filename: str = '',
                    write_pickle: bool = True) -> List[nx.Graph]:

    # make_dirs(outdir=outdir, name=name)
    # if fancy and grammar_type == 'AVRG': grammar_type += '-fancy'
    # if alpha is not None: grammar_type += f'-{int(alpha * 100)}'
    if graphs_filename == '':
        graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'
    gen_filename = f'{outdir}/output/generators/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'

    if use_pickle and check_file_exists(graphs_filename):
        if not save_snapshots:
            logging.error(f'Graph pickle found! {graphs_filename!r}')
            return
        if save_snapshots and check_file_exists(gen_filename):
            logging.error(f'Gen pickle found, skipping: {gen_filename!r}')
            return

    logging.error(f'Graphs filename: {graphs_filename!r}')

    if isinstance(grammar, AttributedVRG):
        assert attr_name != '' and fancy is not None
        if 'greedy' in grammar_type:
            assert inp_attr_ast is not None and inp_deg_ast is not None
            if alpha is None: alpha = 0.5
            gen = GreedyAttributeRandomGenerator(grammar=grammar, mixing_dict=mixing_dict, attr_name=attr_name,
                                                 inp_attr_ast=inp_attr_ast, inp_deg_ast=inp_deg_ast,
                                                 save_snapshots=save_snapshots, alpha=alpha)
        else:
            gen = AttributedRandomGenerator(grammar=grammar, mixing_dict=mixing_dict, attr_name=attr_name,
                                            use_fancy_rewiring=fancy, save_snapshots=save_snapshots)
    elif isinstance(grammar, VRG):
        gen = RandomGenerator(grammar=grammar, save_snapshots=save_snapshots)
    elif isinstance(grammar, NCE):
        gen = NCEGenerator(grammar=grammar)
    else:
        raise NotImplementedError(f'Invalid grammar type {type(grammar)!r}')

    graphs = gen.generate(num_graphs=num_graphs)
    if write_pickle: dump_pickle(graphs, graphs_filename)
    if save_snapshots: dump_pickle(gen, gen_filename)

    return graphs


def parse_args():
    clustering_algs = ['consensus', 'leiden', 'louvain', 'spectral', 'cond', 'random']
    # grammar_types = ('mu_random', )  # 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    grammar_types = ('VRG', 'NCE')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter class shows defaults in help

    # using choices we can control the inputs. metavar='' prevents printing the choices in the help preventing clutter
    parser.add_argument('-g', '--graph', help='Name of the graph or path to gpickle file', default='karate', metavar='')
    parser.add_argument('-c', '--clustering', help='Clustering method to use', default='consensus', choices=clustering_algs,
                        metavar='')
    parser.add_argument('-m', '--mu', help='Size of RHS (mu)', default=4, type=int)
    parser.add_argument('-t', '--type', help='Grammar type', default='VRG-mu_random', choices=grammar_types, metavar='')
    parser.add_argument('-o', '--outdir', help='Name of the output directory', default='output')
    parser.add_argument('-n', help='Number of graphs to generate', default=5, type=int)
    parser.add_argument('-p', '--grammar-pickle', help='Use pickled grammar?', action='store_true')
    parser.add_argument('-d', '--cluster-pickle', help='Use pickled dendrogram?', action='store_true')
    parser.add_argument('-a', '--attr-name', help='Name of Attribute', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    basedir = '/data/ssikdar/Attributed-VRG'
    name = 'polblogs'
    g, att_name = get_graph(gname=name, basedir=basedir)
    exit(1)


    # name = 'karate'; attr_name = 'club'
    mu = 5; grammar_type = ('AVRG', 'all_tnodes')
    use_grammar_pickle = True; use_cluster_pickle = True; n = 10
    inp_deg_ast, inp_attr_ast = None, None

    g, attr_name = get_graph(name, basedir=basedir)
    clustering = 'leiden'
    g.name = name
    if attr_name != '':
        mix_dict = get_mixing_dict(g, attr_name=attr_name)
        print('Mixing dict:', mix_dict)
    else:
        mix_dict = None

    vrg = get_grammars(name=name, clustering=clustering, grammar_type=grammar_type, input_graph=g, mu=mu,
                       use_grammar_pickle=use_grammar_pickle, use_cluster_pickle=use_cluster_pickle, attr_name=attr_name,
                       outdir=basedir)[0]

    print(vrg)

    inp_deg_ast = nx.degree_assortativity_coefficient(g)
    inp_attr_ast = nx.attribute_assortativity_coefficient(g, attr_name)
    grammar_type = 'AVRG-greedy'
    alpha = 0.5
    graphs = generate_graphs(name=name, grammar=vrg, num_graphs=10, mixing_dict=mix_dict, attr_name=attr_name,
                             grammar_type=grammar_type, inp_deg_ast=inp_deg_ast, inp_attr_ast=inp_attr_ast, fancy=False,
                             save_snapshots=False, alpha=alpha)
    print(graphs)
