import random
import numpy as np
import pyintergraph as pig
import graph_tool.all as gt
import networkx as nx
from typing import List
import sys; sys.path.extend(['../', '../../', '../../', '../../../'])

from other_models.AGM.AGM_Generator import agm_fcl_runner


def get_graphs_from_models(input_graph: nx.Graph, model: str, num_graphs: int) -> List[nx.Graph]:

    name = input_graph.name
    assert model in ('DC-SBM', 'CL', 'AGM')
    gt_g = pig.nx2gt(input_graph)

    graphs = []
    for i in range(num_graphs):
        if model == 'CL':
            graph = agm_fcl_runner(input_graph, model='FCL')
        elif model == 'AGM':
            graph = agm_fcl_runner(input_graph, model='AGM')
        elif model == 'DC-SBM':
            graph_gt = dc_sbm(gt_g)
            graph = pig.gt2nx(graph_gt)
        else:
            raise NotImplementedError(f'Invalid model: {model!r}')

        graph.remove_edges_from(nx.selfloop_edges(graph))
        if not nx.is_connected(g):
            nodes_lcc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(nodes_lcc).copy()

        graph.name = f'{model}-{name}-{i+1}'
        graphs.append(graph)

    return graphs


def dc_sbm(gt_g):
    assert isinstance(gt_g, gt.Graph)
    g = gt.GraphView(gt_g, vfilt=gt.label_largest_component(gt_g))
    g = gt.Graph(g, prune=True)
    g.set_directed(False)

    state = gt.minimize_blockmodel_dl(g)

    u = gt.generate_sbm(state.b.a, gt.adjacency(state.get_bg(), state.get_ers()).T,
                        g.degree_property_map("total").a,
                        g.degree_property_map("total").a, directed=False)
    return u


def shuffle_graph(input_g: nx.Graph, num_graphs: int = 10):
    input_g = pig.nx2gt(input_g)
    name = input_g.name
    models = {'Erdos-Renyi': 'erdos', 'CL': 'configuration', 'CL-deg': 'constrained-configuration',
              'CL-attr': 'constrained-configuration'}
    shuffled_graphs = {model: [] for model in models}
    n_iters = np.linspace(0, input_g.num_edges(), num_graphs, endpoint=True, dtype=int)

    for model, m in models.items():
        for n_iter in n_iters:
            new_g = input_g.copy()

            if model == 'CL-attr':
                gt.random_rewire(g=new_g, model=m, n_iter=n_iter, edge_sweep=False, block_membership=new_g.vp.value)
            else:
                gt.random_rewire(g=new_g, model=m, n_iter=n_iter, edge_sweep=False)
            shuffled_graphs[model].append(new_g)
    return shuffled_graphs
