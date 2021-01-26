import random
import sys
from glob import glob
from os.path import join
from pathlib import Path

import networkx as nx
import numpy as np
import os

sys.path.extend(['/home/ssikdar/tmp_dir', '../', '../../', '../../../'])

from VRG.runner import get_grammars, get_graph, get_clustering, generate_graphs, make_dirs
from VRG.src.VRG import AttributedVRG, VRG
from VRG.src.parallel import parallel_async
from VRG.src.utils import load_pickle, get_mixing_dict


def read_batched_graphs(outdir, name):
    input_graphs = load_pickle(join(outdir, 'input', f'{name}.graphs'))
    cleaned_graphs = []

    for i, g in enumerate(input_graphs):
        g.remove_edges_from(nx.selfloop_edges(g))
        if not nx.is_connected(g):
            nodes_lcc = max(nx.connected_components(g), key=len)
            g = g.subgraph(nodes_lcc).copy()
        g = nx.convert_node_labels_to_integers(g, label_attribute='orig_label')
        g.name = f'{name}_{i}'
        cleaned_graphs.append(g)

    return cleaned_graphs


def batch_cluster_shuffler_runner():
    shuffle_kind = 'edges'
    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random', 'consensus']
    clusterings = ['cond', 'leiden', 'louvain', 'leadingeig']
    use_pickle = True
    args = []

    # for graph_filename in glob(f'./input/shuffled/{shuffle_kind}/toy-comm-*.gexf'):
    shuffle_kind = 'attrs'
    for graph_filename in glob(f'./input/shuffled/{shuffle_kind}/toy-comm-0.gexf'):
        path = Path(graph_filename)
        g = nx.read_gexf(graph_filename, node_type=int)
        # name = f'{path.stem}-{shuffle_kind}'
        name = 'toy-comm-attr'
        g.name = name
        for clustering in clusterings:
            args.append((g, f'/data/ssikdar/attributed-vrg/dumps/trees/{name}', clustering, use_pickle))

    parallel_async(func=get_clustering, args=args)
    return


def batch_cluster_runner(names, outdir, clusterings=None):
    if clusterings is None:
        clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random',
                       'leading_eig', 'consensus'][: -1]
    use_pickle = True
    args = []

    for name in names:
        g, _ = get_graph(name, basedir=outdir)
        g.name = name
        for clustering in clusterings:
            args.append((g, outdir, clustering, use_pickle, ''))
    random.shuffle(args)
    parallel_async(func=get_clustering, args=args)
    return


def batch_grammar_runner(names, clusterings, outdir):
    grammar_types_1 = ['VRG', 'AVRG']
    grammar_types_2 = ['mu_random', 'mu_level', 'all_tnodes']
    mus = range(3, 11)
    # mus = [5, 6]
    use_cluster_pickle = True
    use_grammar_pickle = True
    count = 1
    args = []
    write_pickle = True

    for name in names:
        input_graph, attr_name = get_graph(name, basedir=outdir)
        # input_graph, attr_name = nx.read_gexf(f'./input/shuffled/{shuffle}/{name}.gexf', node_type=int), 'block'
        # input_graph, attr_name, name = nx.read_gexf(f'./input/shuffled/{shuffle}/toy-comm-0.gexf', node_type=int), 'block', f'{name}-{shuffle}'

        for clustering in clusterings:
            for grammar_type_1 in grammar_types_1:
                for grammar_type_2 in grammar_types_2:
                    grammar_type = (grammar_type_1, grammar_type_2)
                    for mu in mus:
                        grammar_filename = join(outdir, 'output', 'grammars', name,
                                                f'{grammar_type_1}_{grammar_type_2.replace("_", "-")}_{clustering}_{mu}.pkl')

                        if grammar_type_2 == 'all_tnodes':
                            mu = -1

                        arg = (name, clustering, grammar_type, mu, input_graph, use_grammar_pickle,
                               use_cluster_pickle, attr_name, outdir, count, grammar_filename, write_pickle)
                        args.append(arg)
                        if grammar_type_2 == 'all_tnodes':  # here mu is not important for all_tnodes
                            break
    print(args[: 3])
    random.shuffle(args)
    try:
        parallel_async(func=get_grammars, args=args, num_workers=8)
    except Exception as e:
        print(e)
    return

    # get_grammars(name: str, clustering: str, grammar_type: Tuple[str, str], mu: int, input_graph: nx.Graph,
    #                  use_grammar_pickle: bool, use_cluster_pickle: bool, attr_name: str, count: int = 1)


def batch_generator_runner(names, outdir, clusterings=None, save_snapshots=True):
    num_graphs = 10  # we need 1 graph to chart the progress  # TODO: change this in the future?
    use_pickle = True
    save_snapshots = save_snapshots
    mus = list(range(3, 11)) + [-1]
    # mus = range(5, 8)
    alpha = None
    attr_name = 'value'

    if clusterings is None:  clusterings = ['leiden', 'louvain', 'cond']

    args = []
    for name in names:
        input_graph, attr_name = get_graph(name, basedir=outdir)
        if input_graph.size() > 10_000:
            save_snapshots = False

        mix_dict = get_mixing_dict(input_graph, attr_name=attr_name)
        inp_deg_ast = nx.degree_assortativity_coefficient(input_graph)
        inp_attr_ast = nx.attribute_assortativity_coefficient(input_graph, attr_name)

        for grammar_filename in glob(f'{outdir}/output/grammars/{name}/*'):
            grammar = load_pickle(grammar_filename)
            if grammar.mu not in mus or grammar.clustering not in clusterings:
                continue
            if isinstance(grammar, AttributedVRG):
                grammar_type = 'AVRG-fancy'
                fancy = True
                graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))

                for alpha in (0, 0.5, 1):
                    grammar_type = f'AVRG-greedy-{int(alpha * 100)}'
                    graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))

            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))

    random.shuffle(args)
    try:
        parallel_async(func=generate_graphs, args=args, num_workers=10)
    except Exception as e:
        print(e)
    return


def batch_synthetic_generator_runner():
    # frac = np.linspace(0, 1, 21, endpoint=True) * 100
    frac = np.linspace(0, 100, 11, endpoint=True, dtype=int)  # change it to increments of 10 for now
    names = [f'toy-comm-{f}' for f in frac]
    # names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    num_graphs = 5
    outdir = '/data/ssikdar/attributed-vrg/dumps'
    use_pickle = True
    save_snapshots = False
    shuffle = 'edges'

    args = []
    for name in names:
        # input_graph, attr_name = get_graph(name)
        input_graph, attr_name = nx.read_gexf(f'./input/shuffled/{shuffle}/{name}.gexf', node_type=int), 'block'
        name = f'{name}-{shuffle}'
        if attr_name == '':
            mix_dict, inp_deg_ast, inp_attr_ast = None, None, None
        else:
            mix_dict = get_mixing_dict(input_graph, attr_name=attr_name)
            inp_deg_ast = nx.degree_assortativity_coefficient(input_graph)
            inp_attr_ast = nx.attribute_assortativity_coefficient(input_graph, attr_name)

        for grammar_filename in glob(f'{outdir}/grammars/{name}/*'):
            grammar = load_pickle(grammar_filename)
            if isinstance(grammar, AttributedVRG):
                grammar_type = 'AVRG'
                fancy = True
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))

                grammar_type = 'AVRG-greedy'
                # args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                #              inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))
                for alpha in (0, 0.5, 1):
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha))
            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))

    parallel_async(func=generate_graphs, args=args, num_workers=10)
    # generate_graphs(grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, grammar_type: str, outdir: str = 'dumps',
    #                 mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, fancy = None,
    #                 inp_deg_ast: float = None, inp_attr_ast: float = None)

    return


def batch_synthetic_generator_runner_attrs():
    # frac = np.linspace(0, 1, 21, endpoint=True) * 100
    frac = np.linspace(0, 100, 11, endpoint=True, dtype=int)  # change it to increments of 10 for now
    names = [f'toy-comm-{f}' for f in frac]
    # names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    num_graphs = 5
    outdir = '/data/ssikdar/attributed-vrg/dumps'
    use_pickle = True
    save_snapshots = False
    shuffle = 'attrs'

    args = []
    # input_graph, attr_name = nx.read_gexf(f'./input/shuffled/attrs/toy-comm-0.gexf', node_type=int), 'block'
    attr_name = 'block'
    for f in frac:
        g = nx.read_gexf(f'./input/shuffled/attrs/toy-comm-{f}.gexf', node_type=int)
        mix_dict = get_mixing_dict(g, attr_name=attr_name)
        inp_deg_ast = nx.degree_assortativity_coefficient(g)
        inp_attr_ast = nx.attribute_assortativity_coefficient(g, attr_name)
        name = f'toy-comm-attrs-{f}'

        for grammar_filename in glob(f'{outdir}/grammars/toy-comm-{f}-attrs/*'):
            grammar = load_pickle(grammar_filename)
            if isinstance(grammar, AttributedVRG):
                grammar_type = 'AVRG'
                fancy = True
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))
                grammar_type = 'AVRG-greedy'
                for alpha in (0, 0.5, 1):
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha))
            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))

    parallel_async(func=generate_graphs, args=args, num_workers=13)
    # generate_graphs(grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, grammar_type: str, outdir: str = 'dumps',
    #                 mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, fancy = None,
    #                 inp_deg_ast: float = None, inp_attr_ast: float = None)

    return


def batched_graphs_clusters(outdir, name, clusterings):
    input_graphs = read_batched_graphs(outdir=outdir, name=name)
    use_pickle = True
    args = []

    for i, g in enumerate(input_graphs):
        g.name = f'{name}-{i}'
        for clustering in clusterings:
            filename = os.path.join(outdir, 'output', 'trees', name, f'{clustering}_{i}.pkl')
            args.append((g, f'dumps/trees/{name}', clustering, use_pickle, filename))

    parallel_async(func=get_clustering, args=args)
    return


def batched_graphs_grammars(outdir, name, clusterings):
    input_graphs = read_batched_graphs(outdir=outdir, name=name)
    attr_name = 'value'
    grammar_types_1 = ['VRG', 'AVRG']
    grammar_types_2 = ['mu_random', 'mu_level', 'all_tnodes']
    mus = [5, 6]
    use_cluster_pickle = True
    use_grammar_pickle = True
    count = 1

    args = []
    for i, input_graph in enumerate(input_graphs):
        for clustering in clusterings:
            list_of_list_clusters = load_pickle(join(outdir, 'output', 'trees', name, f'{clustering}_{i}.pkl'))
            for grammar_type_1 in grammar_types_1:
                for grammar_type_2 in grammar_types_2:
                    grammar_type = (grammar_type_1, grammar_type_2)
                    for mu in mus:
                        grammar_filename = f'{outdir}/output/grammars/{name}/{grammar_type[0]}-{grammar_type[1].replace("_", "-")}_{clustering}_{mu}_{i}.pkl'

                        arg = (name, clustering, grammar_type, mu, input_graph, use_grammar_pickle,
                               use_cluster_pickle, attr_name, outdir, count, grammar_filename, True, list_of_list_clusters)
                        args.append(arg)
                        if grammar_type_2 == 'all_tnodes':  # here mu is not important for all_tnodes
                            break
    # print(args[: 3])

    try:
        parallel_async(func=get_grammars, args=args, num_workers=10)
    except Exception as e:
        print(e)
    return


def batched_graphs_generator(outdir, clusterings, name):
    num_graphs = 5 if 'polblogs' in name else 10
    use_pickle = True
    save_snapshots = False
    attr_name = 'value'
    mus = [5, 6, 7]
    alpha = None
    input_graphs = read_batched_graphs(outdir=outdir, name=name)

    args = []
    for i, input_graph in enumerate(input_graphs):
        mix_dict = get_mixing_dict(input_graph, attr_name=attr_name)
        inp_deg_ast = nx.degree_assortativity_coefficient(input_graph)
        inp_attr_ast = nx.attribute_assortativity_coefficient(input_graph, attr_name)

        for grammar_filename in glob(f'{outdir}/output/grammars/{name}/*_{i}.pkl'):
            grammar = load_pickle(grammar_filename)
            if grammar.mu not in mus or grammar.clustering not in clusterings:
                continue
            if isinstance(grammar, AttributedVRG):
                grammar_type = 'AVRG-fancy'
                fancy = True
                graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}_{i}.pkl'
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))

                for alpha in (0, 0.5, 1):
                    grammar_type = f'AVRG-greedy-{int(alpha * 100)}'
                    graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}_{i}.pkl'
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))

            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}_{i}.pkl'
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))

    # random.shuffle(args)
    parallel_async(func=generate_graphs, args=args, num_workers=8)
    return


if __name__ == '__main__':
    outdir = '/data/ssikdar/Attributed-VRG'

    names = ['karate', 'football', 'polbooks', 'wisconsin', 'texas', 'cornell', 'airports',
             'polblogs', 'cora', 'citeseer', 'film', 'chameleon', 'squirrel', 'pubmed']

    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random',
                   'leadingeig', 'consensus'][: -1]

    for name in names:
        make_dirs(outdir=join(outdir, 'output'), name=name)
    # batch_cluster_runner(names=names, clusterings=clusterings, outdir=outdir)
    # batch_grammar_runner(names=names, clusterings=clusterings, outdir=outdir)
    batch_generator_runner(outdir=outdir, names=names)
    exit(1)

    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'random']


    # clusterings = ['cond', 'louvain', 'leiden', 'spectral']
    # name = 'polbooks'
    # for kind in ('deg', 'attr')[1: ]:
    #     name_ = f'{name}-{kind}'
    #     # batched_graphs_clusters(outdir, name=name_, clusterings=clusterings)
    #     # batched_graphs_grammars(outdir=outdir, name=name_, clusterings=clusterings)
    #     batched_graphs_generator(outdir=outdir, name=name_, clusterings=clusterings)
    #
    # # batch_synthetic_generator_runner_attrs()
    # # batch_synthetic_generator_runner()

    # names = ['lang-bip']
    # clusterings = ['cond', 'leiden', 'louvain', 'spectral']
    # batch_cluster_shuffler_runner(names=names, clusterings=clusterings)
    # batch_cluster_runner(names=names, clusterings=clusterings, outdir=outdir)

    # batch_generator_runner(names=names, clusterings=clusterings, outdir=outdir)
