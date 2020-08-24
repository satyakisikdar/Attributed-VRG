"""
Preprocess the graph after reading the edge list
This could be done by modifying node and edge attributes
"""
import logging

import networkx as nx


def assign_weights(nx_g: nx.Graph, method: str) -> None:
    assert isinstance(nx_g, nx.Graph), 'passed Graph must be a NetworkX graph'
    if method == 'jaccard':
        wt_method = nx.jaccard_coefficient
    elif method == 'adamic-adar':
        wt_method = nx.adamic_adar_index
    else:
        raise NotImplementedError(f'Invalid method: {method!r}')

    logging.error(f'Assigning weights via {method!r}')

    wts = {(u, v): min(p + 0.001, 1) for u, v, p in wt_method(nx_g, ebunch=nx_g.edges())}

    nx.set_edge_attributes(nx_g, name='wt', values=wts)
    return
