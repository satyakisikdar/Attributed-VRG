"""
Container for different graph stats
"""
import platform
import subprocess as sub
import sys
from collections import Counter, deque
from typing import Dict, Tuple, List, Any

from VRG.src.utils import nx_to_igraph

sys.path.extend(['./../', './../../'])
print('sys path: ', sys.path)
import editdistance as ed
import matplotlib.pyplot as plt
import networkx as nx
# import NetLSD.netlsd as net
import numpy as np
import seaborn as sns
import igraph as ig

# from src.utils import check_file_exists, ColorPrint as CP, dump_pickle

sns.set()
sns.set_style("darkgrid")

class ColorPrint:
    @staticmethod
    def print_red(message, end='\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_green(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_orange(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_blue(message, end='\n'):
        # pass
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_none(message, end='\n'):
        pass
        # sys.stdout.write(message + end)

CP = ColorPrint


class GraphStats:
    """
    GraphStats has methods for finding different statistics for a NetworkX graph
    """
    __slots__ = ['graph', 'ig_graph', 'stats']

    def __init__(self, graph: nx.Graph):
        self.graph: nx.Graph = graph
        self.ig_graph: ig.Graph = None
        self.stats: Dict[str, Any] = {'name': graph.name, 'n': graph.order(), 'm': graph.size()}

    def __str__(self) -> str:
        st = f'"{self.graph.name}" stats:'
        for key, val in self.stats.items():
            if isinstance(val, float):
                val = round(val, 3)
            elif isinstance(val, dict):
                val = list(val.items())[: 3]  # print the first 3 key value pairs
            elif 'numpy' in str(type(val)):
                val = val[: 3]
            st += f'"{key}": {val} '
        return st

    def __getitem__(self, item):
        """
        Allows square bracket indexing for stats - allow for some fuzzy matching
        """
        if item in self.stats:  # the stat has already been calculated
            return self.stats[item]

        # try some fuzzy matching to figure out thde function to call based on the item
        object_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name)) and not method_name.startswith('_')]

        best_match_func = ''
        best_match_score = float('inf')

        for method in object_methods:
            dist = ed.eval(method, item)
            if dist == 0:
                best_match_score = dist
                best_match_func = method
                break

            if dist < best_match_score:
                best_match_score = dist
                best_match_func = method

        assert best_match_func != '', 'edit distance did not work'
        item = best_match_func
        if best_match_score != 0:
            CP.print_orange(
                f'Best matching function found for "{item}": "{best_match_func}()", edit distance: {best_match_score}')

        if best_match_func not in self.stats:
            best_match_func = getattr(self,
                                      best_match_func)  # translates best_match_fun from string to a function object
            best_match_func()  # call the best match function

        assert item in self.stats, f'stat: {item} is not updated after function call'
        return self.stats[item]

    def plot(self, y, ax=None, kind='line', x=None, **kwargs) -> None:
        if isinstance(y, dict):
            lists = sorted(y.items())
            x, y = zip(*lists)
        else:  # if isinstance(x, list) or isinstance(x, np.array):
            x = list(range(len(y)))

        if kind == 'line':
            # plt.plot(x, y, marker='o', linestyle='--')
            sns.lineplot(x, y, marker='o', dashes='--', ax=ax, **kwargs)  # , dashes=True)
        if kind == 'scatter':
            # plt.scatter(x, y, marker='o')
            ax = sns.scatterplot(x, y, ax=ax, **kwargs)

        title = kwargs.get('title', '')
        xlabel = kwargs.get('xlabel', '')
        ylabel = kwargs.get('ylabel', '')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        return ax

    def _make_igraph_if_necessary(self, fname=None):
        if self.ig_graph is None:
            if fname is not None:
                self.ig_graph = ig.Graph.Read_GML(fname)
            else:
                self.ig_graph = nx_to_igraph(self.graph)
        return

    def average_path_length(self) -> float:
        self._make_igraph_if_necessary()
        assert self.ig_graph is not None
        apl = self.ig_graph.average_path_length(directed=False, unconn=True)
        self.stats['average_path_length'] = apl
        return apl

    def average_clustering(self) -> float:
        self._make_igraph_if_necessary()
        assert self.ig_graph is not None
        avg_cl = self.ig_graph.transitivity_avglocal_undirected(mode='zero')
        self.stats['average_clustering'] = avg_cl
        return avg_cl

    def shortest_path_ast(self, alpha: float, fname=None) -> float:
        self._make_igraph_if_necessary(fname=fname)
        assert self.ig_graph is not None

        n = self.ig_graph.vcount()
        d = np.array(self.ig_graph.degree()).reshape(n, 1)
        if 'apsp_matrix' not in self.stats:
            self.stats['apsp_matrix'] = np.array(self.ig_graph.shortest_paths())

        H = self.stats['apsp_matrix']
        H_alpha = np.float_power(H, -alpha)
        np.fill_diagonal(H_alpha, 0)

        h = H_alpha.sum()
        E = H_alpha / h

        q = E.sum(axis=1).reshape(n, 1)
        D_q = np.diag(q.flatten())
        qq_t = np.multiply(q, q.T)

        X = E - qq_t
        Y = D_q - qq_t

        num = np.matmul(np.matmul(d.T, X), d)
        denom = np.matmul(np.matmul(d.T, Y), d)
        shortest_path_ast = (num / denom).item()
        self.stats[f'shortest_path_ast_{int(alpha * 100)}'] = shortest_path_ast
        return shortest_path_ast

    def degree_mixing_dict(self, normalized: bool = True) -> np.array:
        dict = nx.degree_mixing_dict(self.graph, normalized=normalized)
        self.stats['degree_mixing_dict'] = dict
        return dict

    def attr_mixing_dict(self, attr_name: str = 'value', normalized: bool = True) -> np.array:
        dict = nx.attribute_mixing_dict(self.graph, attribute=attr_name, normalized=normalized)
        self.stats['attr_mixing_dict'] = dict
        return dict

    def degree_mixing_matrix(self, normalized: bool = True) -> np.array:
        mat = nx.degree_mixing_matrix(self.graph, normalized=normalized)
        self.stats['degree_mixing_matrix'] = mat
        return mat

    def attr_mixing_matrix(self, attr_name: str = 'value', mapping: dict = None, normalized: bool = True) -> np.array:
        mat = nx.attribute_mixing_matrix(self.graph, attribute=attr_name, mapping=mapping, normalized=normalized)
        self.stats['attr_mixing_matrix'] = mat
        return mat

    def _calculate_robustness_measures(self) -> None:
        """
        Calls the Leiden comms and frac of nodes in giant component methods
        """
        print('Calling number of components, frac of nodes in giant component, leiden alg')
        print('Populates "num_components", "giant_frac", "num_clusters", "modularity" in self.stats')
        self.stats['num_components'] = nx.number_connected_components(self.graph)
        self.giant_component_frac()
        self.leiden_communities()
        self.giant_component_frac()
        return

    def leiden_communities(self) -> Tuple[int, float]:
        """
        Use Leiden alg to find (a) the number of communities and (b) modularity
        """
        self._make_igraph_if_necessary()
        assert self.ig_graph is not None
        partition = self.ig_graph.community_leiden(n_iterations=-1, objective_function='modularity')
        self.stats['num_clusters'] = len(partition)
        self.stats['modularity'] = partition.modularity
        return len(partition), partition.modularity

    def giant_component_frac(self):
        """
        returns the fraction of nodes in the giant connected component
        """
        lcc = max(nx.connected_components(self.graph), key=len)
        frac = len(lcc) / self.graph.order()
        self.stats['giant_frac'] = frac
        return frac

    def adj_eigenvalues(self):
        """
        Returns the eigenvalues of the Adjacency matrix
        :return:
        """
        CP.print_none('Calculating eigenvalues of Adjacency Matrix')

        adj_eigenvalues = nx.adjacency_spectrum(self.graph)
        self.stats['adj_eigenvalues'] = adj_eigenvalues

        return adj_eigenvalues

    def attribute_assortativity(self, attr_name='value') -> float:
        """
        returns the attribute assortativity
        :param attr_name:
        :return:
        """
        attr_ast = nx.attribute_assortativity_coefficient(self.graph, attribute=attr_name)
        self.stats['attribute_assortativity'] = attr_ast
        return attr_ast

    def degree_assortativity(self) -> float:
        """
        Returns the degree assortativity of the network
        :return:
        """
        # self._make_igraph_if_necessary()
        # assert self.ig_graph is not None
        CP.print_none('Calculating Degree Assortativity')

        # assortativity = self.ig_graph.assortativity_degree(directed=False)
        assortativity = nx.degree_assortativity_coefficient(self.graph)
        self.stats['degree_assortativity'] = assortativity

        return assortativity

    def _calculate_all_stats(self):
        """
        Calculate all stats
        """
        CP.print_orange('Calculating all stats')

        object_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name)) and not method_name.startswith('_')]

        for method in object_methods:
            method = getattr(self, method)
            try:
                method()
            except NotImplementedError as e:
                pass

    def closeness_centrality(self) -> Dict[int, float]:
        """
        Closeness centrality
        """
        CP.print_none('Calculating Closeness Centrality')

        closeness = nx.closeness_centrality(self.graph)
        self.stats['closeness_centrality'] = closeness

        return closeness

    def clustering_coefficients_by_degree(self) -> Dict[int, float]:
        """
        Returns the average clustering coefficient by degree
        :return:
        """
        CP.print_none('Calculating Clustering Coefficients and CC by degree')

        clustering_coeffs = nx.clustering(self.graph)
        self.stats['clustering_coeffs'] = clustering_coeffs

        clustering_by_degree = {}  # clustering per degree

        # get the sums
        for node, cc in clustering_coeffs.items():
            deg = self.graph.degree[node]
            if deg not in clustering_by_degree:
                clustering_by_degree[deg] = []
            clustering_by_degree[deg].append(cc)

        avg_clustering_by_degree = {deg: np.mean(ccs) for deg, ccs in clustering_by_degree.items()}
        self.stats['clustering_coefficients_by_degree'] = avg_clustering_by_degree

        return avg_clustering_by_degree

    def component_size_distribution(self) -> List[Tuple[int, float]]:
        """
        Returns the distribution of component sizes and fraction of nodes in each component, largest first
        :return:
        """
        CP.print_none('Calculating Component Size Distribution')

        component_size_ratio_list = [(len(c), len(c) / self.graph.order()) for c in
                                     sorted(nx.connected_components(self.graph),
                                            key=len, reverse=True)]
        self.stats['component_size_distribution'] = component_size_ratio_list

        return component_size_ratio_list

    def degree_centrality(self) -> Dict[int, float]:
        """
        Degree centrality
        """
        CP.print_none('Calculating Degree Centrality')

        degree_centrality = nx.degree_centrality(self.graph)
        self.stats['degree_centrality'] = degree_centrality

        return degree_centrality

    def degree_dist(self, normalized=True) -> Dict[int, float]:
        """
        Returns the degrees counter - keys: degrees, values: #nodes with that degree
        :return:
        """
        CP.print_none('Calculating Degree Distribution')

        degree_seq = sorted(deg for _, deg in self.graph.degree())
        self.stats['degree_seq'] = degree_seq

        degree_counts = Counter(degree_seq)

        if normalized:
            for deg, count in degree_counts.items():
                degree_counts[deg] /= self.graph.order()

        self.stats['degree_dist'] = dict(degree_counts)
        return dict(degree_counts)

    def diameter(self) -> float:
        CP.print_none('Calculating Diameter')
        self._make_igraph_if_necessary()
        assert self.ig_graph is not None

        if self.stats['apsp_matrix'] is None:
            self.stats['apsp_matrix'] = np.array(self.ig_graph.shortest_paths())
        diam = np.max(self.stats['apsp_matrix'])
        self.stats['diameter'] = diam

        return diam

    def effective_diameter(self) -> None:
        """
        Returns the 90% effective diameter of a graph
        :return:
        """
        raise NotImplementedError()

    def k_hop_reach(self) -> np.array:
        """
        Returns the average number of nodes reachable from any node in k-hops
        Two levels of aggregation:
            1. _k_hop_reachability gives the absolute count of nodes reachable within a k-hops from a node
            2. overall_k_hop_dict aggregates the sum of all absolute counts for all nodes
        Normalizing factor: n ** 2 (once for each step)
        Then convert to a cumulative distribution
        :return:
        """
        CP.print_none('Calculating hop-plot')

        overall_k_hop_dict = Counter()

        for node in self.graph.nodes():
            k_hop_dict = self._k_hop_reachability_counter(node)
            overall_k_hop_dict += Counter(k_hop_dict)

        k_hop_vec = np.array([v for k, v in sorted(overall_k_hop_dict.items(), key=lambda x: x[0])])
        k_hop_vec = k_hop_vec / (self.graph.order() ** 2)

        self.stats['k_hop_reach'] = np.cumsum(k_hop_vec)

        return self.stats['k_hop_reach']

    def _k_hop_reachability_counter(self, node) -> Dict[int, float]:
        """
        computes fraction of nodes reachable from the given node in k hops
        :param node: node to compute the k_hop_reach vector
        :return:
        """
        reachability_counter = {0: 1}  # within 0 hops, you can reach 1 node - itself
        hop_counter = {node: 0}  # node is 0 hops away from itself
        queue = deque([node])

        while len(queue) != 0:
            node = queue.popleft()
            for nbr in self.graph.neighbors(node):
                if nbr not in hop_counter:  # unexplored neighbor
                    hop_counter[nbr] = hop_counter[node] + 1  # update hop distance of neighbor

                    if hop_counter[nbr] not in reachability_counter:
                        reachability_counter[hop_counter[nbr]] = 0  # reachability_counter[hop_counter[node]]
                    reachability_counter[hop_counter[nbr]] += 1  # keep track of fraction of nodes reachable

                    queue.append(nbr)

        # normalized_reachability_counter = {key: value / n for key, value in reachability_counter.items()}
        return reachability_counter

    def laplacian_eigenvalues(self) -> np.array:
        """
        Returns eigenvalues of the Laplacian
        :return:
        """
        CP.print_none('Calculating Laplacian Eigenvalues')

        laplacian_eigs = nx.laplacian_spectrum(self.graph)
        self.stats['laplacian_eigenvalues'] = laplacian_eigs

        return laplacian_eigs

    def pagerank(self) -> Dict[int, float]:
        """
        PageRank centrality
        """
        CP.print_none('Calculating PageRank')
        self._make_igraph_if_necessary()
        assert self.ig_graph is not None
        pagerank = {i: pr for i, pr in enumerate(self.ig_graph.pagerank(directed=False))}
        # pagerank = nx.pagerank_scipy(self.graph)
        self.stats['pagerank'] = pagerank

        return pagerank

    def pgd_graphlet_counts(self, n_threads=4) -> Dict:
        """
        Return the dictionary of graphlets and their counts - based on Neville's PGD
        :return:
        """
        pgd_path = './src/PGD'
        graphlet_counts = {}

        if 'Linux' in platform.platform() and check_file_exists(f'{pgd_path}/pgd_0'):
            edgelist = '\n'.join(nx.generate_edgelist(self.graph, data=False))
            edgelist += '\nX'  # add the X
            dummy_path = f'{pgd_path}/dummy.txt'

            try:
                bash_script = f'{pgd_path}/pgd -w {n_threads} -f {dummy_path} -c {dummy_path}'

                pipe = sub.run(bash_script, shell=True, capture_output=True, input=edgelist.encode(), check=True,
                               timeout=30000)  # timeout of heat death of universe

                output_data = pipe.stdout.decode()

            except sub.TimeoutExpired as e:
                CP.print_blue(f'PGD timeout!{e.stderr}')
                graphlet_counts = {}

            except sub.CalledProcessError as e:
                CP.print_blue(f'PGD error {e.stderr}')
                graphlet_counts = {}
            except Exception as e:
                CP.print_blue(str(e))
                graphlet_counts = {}
            else:  # pgd is successfully run
                for line in output_data.split('\n')[: -1]:  # last line blank
                    graphlet_name, count = map(lambda st: st.strip(), line.split('='))
                    graphlet_counts[graphlet_name] = int(count)
        else:
            graphlet_counts = {}
        self.stats['pgd_graphlet_counts'] = graphlet_counts

        return graphlet_counts


if __name__ == '__main__':
    # g = nx.karate_club_graph()
    # g = nx.ring_of_cliques(50, 4)
    g = nx.erdos_renyi_graph(5, 0.2, seed=1)
    # g = nx.path_graph(5)
    gs = GraphStats(graph=g, trial=0, dataset='karate', model='blah', iteration=0)
    print(gs.netlsd())
    print(gs.stats)