import networkx as nx


class LightMultiGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)
        self._succ = self._succ if hasattr(self, "_succ") else self._adj

    def size(self, weight=None):
        return int(super(LightMultiGraph, self).size(weight='weight'))

    def __repr__(self):
        return f'n = {self.order():_d} m = {self.size():_d}'

    def add_edge(self, u, v, attr_dict=None, **attr):
        # print(f'inside add_edge {u}, {v}')
        if attr_dict is not None and 'weight' in attr_dict:
            weight = attr_dict['weight']
        elif attr is not None and 'weight' in attr:
            weight = attr['weight']
        else:
            weight = 1
        if self.has_edge(u, v):  # edge already exists
            # print(f'edge ({u}, {v}) exists, {self[u][v]["weight"]}')
            self[u][v]['weight'] += weight
        else:
            super(LightMultiGraph, self).add_edge(u, v, weight=weight)

    def copy(self):
        g_copy = LightMultiGraph()
        for node, d in self.nodes(data=True):
            if len(d) == 0:  # prevents adding an empty 'attr_dict' dictionary
                g_copy.add_node(node)
            else:
                if 'nt' in d:  # this keeps the label and the b_deg attributes to the same level
                    g_copy.add_node(node, nt=d['nt'])
                g_copy.add_node(node, attr_dict=d)
        for e in self.edges(data=True):
            u, v, d = e
            g_copy.add_edge(u, v, attr_dict=d)
        return g_copy

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        for e in ebunch:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesnt need edge_attr_dict_factory
            else:
                raise nx.NetworkXError("Edge tuple %s must be a 2-tuple or 3-tuple." % (e,))
            self.add_edge(u, v, attr_dict=dd, **attr)

    def number_of_edges(self, u=None, v=None):
        if u is None:
            return self.size()
        try:
            return self[u][v]['weight']
        except KeyError:
            return 0  # no such edge

    def degree_(self, n) -> int:
        """
        Modified version of the degree function - needed because the regular degree() call doesnt include
        the weight of the edges
        :param n:
        :return:
        """
        nbrs = self._succ[n]
        deg = sum(self[n][nbr]['weight'] for nbr in nbrs)
        return deg


if __name__ == '__main__':
    g = LightMultiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 2)
    print(g.degree(1))
    # nt1 = NonTerminal(id=50, size=5, nodes_covered=set())
    # g.add_edge(1, nt1)
    #
    # print(g.edges(data=True))
    # print(g.number_of_edges(1, 2))
    #
    # print(set(g.nodes()) & {1, 2, 3, 50})
    # print(list(g.subgraph([3, 50])))
