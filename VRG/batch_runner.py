import sys
from glob import glob
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.extend(['/home/ssikdar/tmp_dir', '../', '../../', '../../../'])

from VRG.runner import get_grammars, get_graph, get_clustering, generate_graphs
from VRG.src.VRG import AttributedVRG, VRG
from VRG.src.parallel import parallel_async
from VRG.src.utils import load_pickle, get_mixing_dict


def batch_cluster_shuffler_runner():
    shuffle_kind = 'edges'
    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random', 'consensus']
    clusterings = ['cond', 'spectral', 'leiden']
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


def batch_cluster_runner():
    names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random', 'consensus']
    use_pickle = False
    args = []

    for name in names:
        g, _ = get_graph(name)
        for clustering in clusterings:
            args.append((g, f'dumps/trees/{name}', clustering, use_pickle))

    parallel_async(func=get_clustering, args=args)
    return


def batch_grammar_runner():
    frac = np.linspace(0, 100, 11, endpoint=True, dtype=int)
    names = [f'toy-comm-{f}' for f in frac]
    # names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'consensus']
    clusterings = ['cond', 'spectral', 'leiden']
    outdir = '/data/ssikdar/attributed-vrg/dumps/'
    grammar_types_1 = ['VRG', 'AVRG']
    grammar_types_2 = ['mu_random', 'all_tnodes']
    # mus = range(3, 11)
    mus = [5, 6]
    use_cluster_pickle = True
    use_grammar_pickle = True
    count = 1
    shuffle = 'attrs'
    # names = ['toy-comm']
    args = []

    for name in names:
        # input_graph, attr_name = get_graph(name)
        input_graph, attr_name = nx.read_gexf(f'./input/shuffled/{shuffle}/{name}.gexf', node_type=int), 'block'
        # input_graph, attr_name = nx.read_gexf(f'./input/shuffled/{shuffle}/toy-comm-0.gexf', node_type=int), 'block'
        name = f'{name}-{shuffle}'
        for clustering in clusterings:
            if attr_name == '':
                grammar_types_1 = ['VRG']  # , 'NCE']  # no AVRG
            for grammar_type_1 in grammar_types_1:
                for grammar_type_2 in grammar_types_2:
                    grammar_type = (grammar_type_1, grammar_type_2)
                    for mu in mus:
                        arg = (name, clustering, grammar_type, mu, input_graph, use_grammar_pickle,
                                     use_cluster_pickle, attr_name, outdir, count)
                        args.append(arg)
                        if grammar_type_2 == 'all_tnodes':  # here mu is not important for all_tnodes
                            break
    print(args[: 3])

    try:
        parallel_async(func=get_grammars, args=args, num_workers=10)
    except Exception:
        pass
    return

    # get_grammars(name: str, clustering: str, grammar_type: Tuple[str, str], mu: int, input_graph: nx.Graph,
    #                  use_grammar_pickle: bool, use_cluster_pickle: bool, attr_name: str, count: int = 1)


def batch_generator_runner():
    # frac = np.linspace(0, 1, 21, endpoint=True) * 100
    frac = np.linspace(0, 100, 11, endpoint=True, dtype=int)  # change it to increments of 10 for now
    names = [f'3-comm-{int(f)}' for f in frac]
    # names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    num_graphs = 5
    outdir = '/data/ssikdar/attributed-vrg/dumps'
    use_pickle = True
    save_snapshots = True
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
                for fancy in (True, False):
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))
                grammar_type = 'AVRG-greedy'
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))
            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))

    parallel_async(func=generate_graphs, args=args)
    # generate_graphs(grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, grammar_type: str, outdir: str = 'dumps',
    #                 mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, fancy = None,
    #                 inp_deg_ast: float = None, inp_attr_ast: float = None)

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


if __name__ == '__main__':
    batch_synthetic_generator_runner_attrs()
    # batch_synthetic_generator_runner()
    # batch_cluster_shuffler_runner()
    # batch_grammar_runner()
    # batch_generator_runner()
    # batch_cluster_runner()
    # get_grammars(name, clustering, grammar_type, mu, input_graph, use_cluster_pickle, use_grammar_pickle)
