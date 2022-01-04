from copy import deepcopy
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
    if graph.is_coloring_proper(coloring):
        return coloring
    else:
        return None


c = None
while c is None:
    c = try_coloring(g)

print(c)
# g.draw(c)

init_col = c


def monte_carlo(graph: ColoredGraph, n_iter: int, init_coloring: List[int]):
    n_nodes = len(graph.nodes)
    coloring_ = init_coloring
    k = graph.get_max_degree() + 2
    available_colors = [i + 1 for i in range(k)]
    for _ in range(n_iter):
        new_coloring = deepcopy(coloring_)
        vertex_idx = np.random.choice([i for i in range(n_nodes)])
        color = np.random.choice(available_colors)
        new_coloring[vertex_idx] = color
        if graph.is_coloring_proper(new_coloring):
            coloring_ = new_coloring

    return coloring_.tolist()


generated_colorings = [monte_carlo(g, 40, init_col) for _ in range(3000)]
distribution = {}

unique_colorings = []
for coloring in generated_colorings:
    if coloring not in unique_colorings:
        unique_colorings.append(coloring)

for unique_coloring in unique_colorings:
    distribution[str(unique_coloring)] = generated_colorings.count(unique_coloring)

print(len(unique_colorings))
pprint(distribution)

print([str(i) for i in generated_colorings])
plt.hist([str(i) for i in generated_colorings], bins=20)
plt.show()
