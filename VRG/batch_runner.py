import os

from VRG.runner import get_grammars, get_graph, get_clustering
from VRG.src.parallel import parallel_async
from VRG.src.utils import load_pickle


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
    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'consensus', 'infomap', 'labelprop']
    grammar_types = [('VRG', 'random'), ('VRG', 'greedy'), ('AVRG', 'greedy'),]
    mus = range(3, 11)
    use_cluster_pickle = True
    use_grammar_pickle = True
    num_graphs = 10

    args = []
    for name in names[-2:]:
        input_graph, attr_name = get_graph(name)
        if attr_name == '':
            grammar_types_1 = ['VRG']  # , 'NCE']  # no AVRG
            mix_dict = None

        else:
            pass
        for clustering in clusterings:
            for grammar_type_1 in grammar_types_1:
                for grammar_type_2 in grammar_types_2:
                    for mu in mus:
                        for i in range(5):
                            grammar_filename = os.path.join('dumps', 'grammars',
                                                            f'{grammar_type_1}-{grammar_type_2}_{clustering}_{mu}_{i}.pkl')
                            grammar = load_pickle(grammar_filename)

    # generate_graphs(grammar, num_graphs: int, grammar_type: str, outdir: str = 'dumps',
    #                 mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, inp_deg_ast: float = None,
    #                 inp_attr_ast: float = None):

    # inp_deg_ast = nx.degree_assortativity_coefficient(g)
    # inp_attr_ast = nx.attribute_assortativity_coefficient(g, attr_name)
    # grammar_type = ('AVRG', 'greedy')
    # graphs = generate_graphs(grammar=vrg, num_graphs=10, mixing_dict=mix_dict, attr_name=attr_name, grammar_type=grammar_type,
    #                          inp_deg_ast=inp_deg_ast, inp_attr_ast=inp_attr_ast)
    return


if __name__ == '__main__':
    # batch_cluster_runner()
    batch_grammar_runner()
    # get_grammars(name, clustering, grammar_type, mu, input_graph, use_cluster_pickle, use_grammar_pickle)
