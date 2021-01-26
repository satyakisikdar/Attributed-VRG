import logging
import os
import random
from os.path import join
from pathlib import Path

import numpy as np
import networkx as nx
from typing import List
import sys; sys.path.extend(['../', '../../', '../../', '../../../'])
from VRG.src.utils import load_pickle, dump_pickle
from other_models.AGM.AGM_Generator import agm_fcl_runner


def get_graphs_from_models(input_graph: nx.Graph, name: str, model: str, num_graphs: int, outdir: str) -> List[nx.Graph]:
    import pyintergraph as pig
    assert model in ('SBM', 'DC-SBM', 'CL', 'AGM')
    graphs_filename = join(outdir, 'output', 'graphs', name, f'{model}_{num_graphs}.pkl')
    if Path(graphs_filename).exists():
        return load_pickle(graphs_filename)

    gt_g = pig.nx2gt(input_graph)

    graphs = []
    for i in range(num_graphs):
        if model == 'CL':
            graph = agm_fcl_runner(input_graph, model='FCL')
        elif model == 'AGM':
            graph = agm_fcl_runner(input_graph, model='AGM')
        elif 'SBM' in model:
            degree_corr = False if model == 'SBM' else True
            graph_gt = dc_sbm(gt_g, degree_corr=degree_corr)
            graph = pig.gt2nx(graph_gt)
        else:
            raise NotImplementedError(f'Invalid model: {model!r}')

        graph.remove_edges_from(nx.selfloop_edges(graph))
        if not nx.is_connected(graph):
            nodes_lcc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(nodes_lcc).copy()

        graph.name = f'{model}-{name}-{i+1}'
        graphs.append(graph)

    dump_pickle(graphs, graphs_filename)
    return graphs


def dc_sbm(gt_g, degree_corr=True):
    import graph_tool.all as gt
    assert isinstance(gt_g, gt.Graph)
    g = gt.GraphView(gt_g, vfilt=gt.label_largest_component(gt_g))
    g = gt.Graph(g, prune=True)
    g.set_directed(False)

    state = gt.minimize_blockmodel_dl(g, deg_corr=degree_corr)

    u = gt.generate_sbm(state.b.a, gt.adjacency(state.get_bg(), state.get_ers()).T,
                        g.degree_property_map("total").a,
                        g.degree_property_map("total").a, directed=False)
    return u


def shuffle_graph(input_g: nx.Graph, num_graphs: int = 10):
    import graph_tool.all as gt
    import pyintergraph as pig

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


def netgan(input_g: nx.Graph, name: str, outdir: str, num_graphs: int = 10, use_model_pickle: bool = True,
           write_model_pickle: bool = True) -> List[nx.Graph]:
    from other_models.netgan.fit import fit
    from other_models.netgan.gen import generate

    model_path = join(outdir, 'output', 'other_models', 'netgan', f'{name}-model.pkl')
    graphs_path = join(outdir, 'output', 'graphs', name, f'NetGAN_{num_graphs}.pkl')

    model = None
    if use_model_pickle:
        model = load_pickle(model_path)

    if model is None:
        mat = nx.to_scipy_sparse_matrix(input_g, nodelist=sorted(input_g.nodes()))
        model = fit(mat)
        if write_model_pickle: dump_pickle(model, model_path)

    assert model is not None, f'NetGAN did not work for {name!r}'

    graphs = generate(*model, name=name, num_graphs=num_graphs, orig_vals_dict=nx.get_node_attributes(input_g, 'value'))
    dump_pickle(graphs, graphs_path)
    return graphs


def _fit_cell(input_g: nx.Graph, steps: int = 200):
    import torch
    from other_models.CELL.src.cell.cell import Cell, EdgeOverlapCriterion
    train_adj = nx.to_scipy_sparse_matrix(input_g, nodelist=sorted(input_g.nodes()))

    # initialize model with EO-criterion
    model = Cell(A=train_adj,
                 H=9,
                 callbacks=[EdgeOverlapCriterion(invoke_every=10, edge_overlap_limit=.5)])

    # train model
    model.train(steps=steps,
                optimizer_fn=torch.optim.Adam,
                optimizer_args={'lr': 0.1, 'weight_decay': 1e-7})

    return model


def _gen_cell_graphs(model, name, num_graphs, orig_vals_dict=None):
    graphs = []
    for i in range(num_graphs):
        sparse_mat = model.sample_graph()
        g = nx.from_scipy_sparse_matrix(sparse_mat, create_using=nx.Graph())
        g.name = f'{name}-CELL'
        if orig_vals_dict is not None:
            nx.set_node_attributes(g, name='value', values=orig_vals_dict)
        graphs.append(g)
    return graphs


def cell(input_g: nx.Graph, name: str, outdir: str, num_graphs: int = 10, steps: int = 200,
         use_model_pickle: bool = True, write_model_pickle: bool = True) -> List[nx.Graph]:

    model_path = join(outdir, 'output', 'other_models', 'cell', f'{name}-model.pkl')
    graphs_path = join(outdir, 'output', 'graphs', name, f'CELL_{num_graphs}.pkl')

    model = None
    if use_model_pickle:
        model = load_pickle(model_path)

    if model is None:
        model = _fit_cell(input_g=input_g, steps=steps)

    if write_model_pickle: dump_pickle(model, model_path)

    graphs = _gen_cell_graphs(model=model, name=name, num_graphs=num_graphs,
                              orig_vals_dict=nx.get_node_attributes(input_g, 'value'))
    dump_pickle(graphs, graphs_path)
    return graphs
