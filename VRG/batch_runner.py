import sys
from glob import glob

import networkx as nx

from VRG.runner import get_grammars, get_graph, get_clustering, generate_graphs
from VRG.src.VRG import AttributedVRG, VRG
from VRG.src.parallel import parallel_async
from VRG.src.utils import load_pickle, get_mixing_dict

sys.path.extend(['/home/ssikdar/tmp_dir'])


def batch_cluster_runner():
    names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random', 'consensus']
    use_pickle = True
    args = []

    for name in names:
        g, _ = get_graph(name)
        for clustering in clusterings:
            args.append((g, f'dumps/trees/{name}', clustering, use_pickle))

    parallel_async(func=get_clustering, args=args)
    return


def batch_grammar_runner():
    names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'consensus']
    grammar_types_1 = ['VRG', 'AVRG']
    grammar_types_2 = ['mu_random', 'all_tnodes']
    mus = range(3, 11)
    use_cluster_pickle = True
    use_grammar_pickle = True
    count = 5

    args = []
    for name in names:
        input_graph, attr_name = get_graph(name)
        for clustering in clusterings:
            if attr_name == '':
                grammar_types_1 = ['VRG']  # , 'NCE']  # no AVRG
            for grammar_type_1 in grammar_types_1:
                for grammar_type_2 in grammar_types_2:
                    grammar_type = (grammar_type_1, grammar_type_2)
                    for mu in mus:
                        args.append((name, clustering, grammar_type, mu, input_graph, use_grammar_pickle,
                                     use_cluster_pickle, attr_name, count))
                        if grammar_type_2 == 'all_tnodes':  # here mu is not important for all_tnodes
                            break
    print(args[: 3])
    try:
        parallel_async(func=get_grammars, args=args)
    except Exception:
        pass
    return

    # get_grammars(name: str, clustering: str, grammar_type: Tuple[str, str], mu: int, input_graph: nx.Graph,
    #                  use_grammar_pickle: bool, use_cluster_pickle: bool, attr_name: str, count: int = 1)


def batch_graph_runner():
    names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    num_graphs = 10
    outdir = 'dumps'
    use_pickle = True

    args = []
    for name in names[2: ]:
        input_graph, attr_name = get_graph(name)
        if attr_name == '':
            mix_dict, inp_deg_ast, inp_attr_ast = None, None, None
        else:
            mix_dict = get_mixing_dict(input_graph, attr_name=attr_name)
            inp_deg_ast = nx.degree_assortativity_coefficient(input_graph)
            inp_attr_ast = nx.attribute_assortativity_coefficient(input_graph, attr_name)

        for grammar_filename in glob(f'./dumps/grammars/{name}/*'):
            grammar = load_pickle(grammar_filename)
            if isinstance(grammar, AttributedVRG):
                grammar_type = 'AVRG'
                for fancy in (True, False):
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle))
                grammar_type = 'AVRG-greedy'
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle))
            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle))

    parallel_async(func=generate_graphs, args=args)
    # generate_graphs(grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, grammar_type: str, outdir: str = 'dumps',
    #                 mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, fancy = None,
    #                 inp_deg_ast: float = None, inp_attr_ast: float = None)

    return


if __name__ == '__main__':
    batch_graph_runner()
    # batch_cluster_runner()
    # batch_grammar_runner()
    # get_grammars(name, clustering, grammar_type, mu, input_graph, use_cluster_pickle, use_grammar_pickle)
