from pprint import pprint
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from matplotlib import colors as mcolors


class ColoredGraph(nx.Graph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.color_names = [k for k, _ in mcolors.cnames.items()][10:]

    def get_adj_matrix(self) -> np.ndarray:
        return nx.linalg.adjacency_matrix(self).toarray()

    def is_coloring_proper(self, colors_list: List[int]) -> bool:
        adj_matrix = self.get_adj_matrix()
        n_nodes = len(self.nodes)
        if len(colors_list) != n_nodes:
            raise ValueError("The number of nodes is not the same as the length of coloring list.")

        if 0 in colors_list:
            raise ValueError("colors_list must be a list of non zero integers.")

        # 200 IQ
        for i in range(n_nodes):
            color = colors_list[i]
            v = adj_matrix[i, :]
            v = v * colors_list - np.ones(n_nodes) * color
            if np.any(v == 0):
                return False

        return True

    def draw(self, colors_list: List[int]):
        node_color = [self.color_names[i] for i in colors_list]
        nx.draw_networkx(self, node_color=node_color)
        plt.show()

    def get_max_degree(self):
        return max(i[1] for i in self.degree)


g = ColoredGraph()

g.add_nodes_from([i + 1 for i in range(4)])
g.add_edge(1, 2)
g.add_edge(2, 4)
g.add_edge(1, 4)
g.add_edge(3, 4)
# g.add_edge(1, 2)
# g.add_edge(1, 3)
# g.add_edge(1, 4)
# g.add_edge(1, 5)
# g.add_edge(2, 4)
# g.add_edge(3, 5)
# g.add_edge(5, 6)
# g.add_edge(7, 3)
# g.add_edge(7, 6)


def try_coloring(graph: ColoredGraph):
    k = graph.get_max_degree() + 2
    coloring = np.random.choice([i + 1 for i in range(k)], len(graph.nodes))
    graph.draw(coloring)
    return graph.is_coloring_proper(coloring)


n = 10000
proper_colorings = []


print(try_coloring(g))
