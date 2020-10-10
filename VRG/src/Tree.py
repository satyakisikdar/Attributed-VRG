from collections import deque
from typing import Tuple, List, Set, Union, Any

import networkx as nx
import numpy as np
from anytree import Node, RenderTree
from joblib import Parallel, delayed

from VRG.src.utils import timer


def get_leaves(tnode) -> Set[int]:
    """
    Returns the set of leaves rooted at tnode
    :return:
    """
    return set(leaf.name for leaf in tnode.leaves)  # get the set of leaf nodes - actual nodes in graph


class TreeNode(Node):
    def __init__(self, name, score=None, **kwargs):
        super().__init__(name, **kwargs)
        self.score = score  # score of that tree node

    def __str__(self) -> str:
        if self.parent is None:
            parent = None
        else:
            parent = self.parent.name
        return f'{self.name} ({len(self.leaves)}) p: {parent} score: {self.score}'


class TreeNodeOld:
    """
    Node class for trees
    """
    __slots__ = 'key', 'level', 'children', 'leaves', 'parent', 'kids', 'is_leaf'

    def __init__(self, key: str, is_leaf: bool=False) -> None:
        self.key = key   # key of the node, each node has an unique key
        self.level = 0  # level of the node

        self.children: Set[Union[int, str]] = set()  # set of node labels of nodes in the subtree rooted at the node
        self.leaves: Set[int] = set()  # set of children that are leaf nodes

        self.parent: Union[TreeNode, None] = None  # pointer to parent
        self.kids: List[TreeNode] = []  # pointers to the children

        self.is_leaf = is_leaf  # True if it's a child, False otherwise

    def __eq__(self, other) -> bool:
        return self.key == other

    def __str__(self) -> str:
        if self.parent is None:
            parent = None
        else:
            parent = self.parent.key

        return f'{self.key} ({len(self.leaves)}) p: {parent}'

    def __repr__(self) -> str:
        return f'{self.key} ({len(self.leaves)})'

    def __copy__(self):
        node_copy = TreeNode(key=self.key)

        node_copy.parent = self.parent
        node_copy.kids = self.kids

        node_copy.leaves = self.leaves
        node_copy.children = self.children

        node_copy.level = self.level
        node_copy.is_leaf = self.is_leaf

        return node_copy

    def __hash__(self):
        return hash(self.key)

    def copy(self):
        return self.__copy__()

    def make_leaf(self, new_key) -> None:
        """
        converts the internal tree node into a leaf
        :param new_key: new key of the node
        :return:
        """
        self.leaves = {self.key}  # update the leaves
        self.children = set()
        self.kids = []
        self.is_leaf = True
        self.key = new_key

    def get_num_leaves(self) -> int:
        return len(self.leaves)


name = 'a'  # needs to be global to make sure the nodes are named appropriately
def create_tree(lst: List) -> TreeNode:
    """
    Creates a AnyTree treenode
    :param lst:
    :return:
    """
    global name
    if len(lst) == 1 and (isinstance(lst[0], int) or isinstance(lst[0], np.int32)):  # leaf
        tnode = TreeNode(name=lst[0])
        return tnode

    tnode = TreeNode(name=name)
    name = chr(ord(name) + 1)

    for sub_lst in lst:
        child_tnode = create_tree(sub_lst)
        child_tnode.parent = tnode

    return tnode


def create_tree_old(lst: List[Any]) -> TreeNode:
    """
    Creates a Tree from the list of lists
    :param lst: nested list of lists
    :return: root of the tree
    """
    key = 'a'

    def create(lst):
        nonlocal key

        if len(lst) == 1 and isinstance(lst[0], int):  # detect leaf
            return TreeNode(key=lst[0], is_leaf=True)
        node = TreeNode(key=key)
        key = chr(ord(key) + 1)

        for item in lst:
            node.kids.append(create(item))

        return node

    root = create(lst)

    def update_info(node) -> Tuple[Set[Union[int, str]], int]:
        """
        updates the parent pointers, payloads, and the number of leaf nodes
        :param node:
        :return:
        """
        if node.is_leaf:
            node.make_leaf(new_key=node.key)  # the key doesn't change
        else:
            for kid in node.kids:
                kid.parent = node
                kid.level = node.level + 1
                children, leaves = update_info(kid)
                node.children.add(kid.key)
                node.children.update(children)
                node.leaves.update(leaves)

        return node.children, node.leaves

    def relabel_tnodes(tnode):
        q = deque()
        q.append(tnode)
        key = 'a'
        while len(q) != 0:
            tnode = q.popleft()
            if tnode.is_leaf:
                continue
            tnode.key = key
            key = chr(ord(key) + 1)
            for kid in tnode.kids:
                q.append(kid)

    relabel_tnodes(root)
    update_info(node=root)

    return root


def find_lca(tnode1: TreeNode, tnode2: TreeNode) -> TreeNode:
    """
    Find the LCA of the two tree nodes
    """
    path1, path2 = map(reversed, (tnode1.path, tnode2.path))  # reverse the paths
    path1, path2 = map(list, (path1, path2))
    higher_depth = tnode1.depth if tnode1.depth < tnode2.depth else tnode2.depth
    path1 = path1[-higher_depth: ]  # only look at the
    path2 = path2[-higher_depth: ]
    for anc1, anc2 in zip(path1, path2):
        if anc1.name == anc2.name:
            lca = anc1
            break
    return lca


@timer
def dasgupta_cost(g: nx.Graph, root: TreeNode, use_parallel: bool = True) -> float:
    """
    Assumes that the graph is unweighted for now
    :param g:
    :param root:
    :param weighted:
    :return:
    """
    tnodes = {leaf.name: leaf for leaf in root.leaves}

    if use_parallel:
        with Parallel(n_jobs=10, backend='multiprocessing') as parallel:
            cost = parallel(delayed(dasgupta_parallel)(u, v, tnodes) for u, v in g.edges())
            total_cost = sum(cost)
    else:
        total_cost = sum(dasgupta_parallel(u, v, tnodes) for u, v in g.edges())
    print(f'Dasgupta cost: {total_cost:_d} parallel? {use_parallel}')
    return total_cost


def dasgupta_parallel(u, v, tnodes) -> float:
    w = 1  # unweighted
    lca_uv = find_lca(tnodes[u], tnodes[v])
    return w * len(lca_uv.leaves)


if __name__ == '__main__':
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

    root = create_tree(list_of_list_clusters)
    tnode_d = root.children[0].children[1]
    tnode_d.children = tuple()
    tnode_d.name = 'nt1'
    print(RenderTree(root))
