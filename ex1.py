from copy import deepcopy
from pprint import pprint
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from matplotlib import colors as mcolors
from scipy.special import factorial
import math


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


# generated_colorings = [monte_carlo(g, 40, init_col) for _ in range(3000)]
# distribution = {}
#
# unique_colorings = []
# for coloring in generated_colorings:
#     if coloring not in unique_colorings:
#         unique_colorings.append(coloring)
#
# for unique_coloring in unique_colorings:
#     distribution[str(unique_coloring)] = generated_colorings.count(unique_coloring)
#
# print(len(unique_colorings))
# pprint(distribution)
#
# print([str(i) for i in generated_colorings])
# plt.hist([str(i) for i in generated_colorings], bins=20)
# plt.show()


def a(i, j, lambda_=1):
    return math.pow(lambda_, j - i) * math.factorial(i) / math.factorial(j)


def T(i, j):
    if i == 0 and j in [0, 1]:
        return 1 / 2
    elif j in [i - 1, i + 1]:
        return 1 / 2
    else:
        return 0.0


for i in range(10):
    for j in range(11):
        print(T(i, j), end=' ')
    print()

for i in range(4):
    for j in range(4):
        print(f"a({i}, {j}) = {a(i, j)}")

# i = 5
# 1/2 szansy, że j = 4 i 1/2 że j = 6
# wylosowano j = 4
#   a(5, 4) >= 1, wiec przechodzimy do stanu 4
#       P(X_n+1 = 4) = 1/2
# wylosowano j = 6
#   przypadek, gdzie idzemy do j
#       P(X_n+1 = 6) = 1/2 a(5, 6)
#   przypadek, gdzie zostajemy w i
#       P(X_n+1 = 5) = 1/2 (1 - a(5, 6))


# print(a(2,3))
# print(a(3,2))
print(a(5, 4))


def metropolis_hastings_poisson_distribution(lambda_, n_iter, i=0):
    for _ in range(n_iter):
        if i == 0:
            j = np.random.choice([0, 1])
        else:
            j = np.random.choice([i - 1, i + 1])

        a_ = a(i, j, lambda_=lambda_)

        if a_ >= 1.0:
            i = j
        else:
            i = np.random.choice([j, i], p=[a_, 1 - a_])

    return i


samp = [metropolis_hastings_poisson_distribution(10, 100, i=0) for _ in range(5000)]
plt.hist(samp, bins=20, density=True)

t = np.arange(0, 20, 0.1)
d = np.exp(-10) * np.power(10, t) / factorial(t)

plt.plot(t, d, 'bs')
plt.show()
