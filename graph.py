from typing import List, Tuple
from timing import timeit
import cv2
import numpy as np

INF = 99999999


def f(x: np.ndarray, y: np.ndarray) -> np.float:
    return np.float(np.exp(-np.sqrt((x - y) @ (x - y))) * 1000)


class Arc:
    def __init__(self, u: int, v: int):
        self.u: int = u
        self.v: int = v
        self.capacity: float = .0
        self.flow: float = .0
        self.is_residual: bool = False
        self.residual = None


class Network:
    def __init__(self):
        self.nodes_count: int = 0
        self.nodes: List[int] = []
        self.arcs: List[Arc] = []
        self.residual_arcs: List[Arc] = []
        self.adj_list: List[List[Arc]] = []
        self.residual_adj_list: List[List[Arc]] = []
        self.adj_matrix: List[List[Arc]] = []

        self.photo_width: int = 0
        self.photo_height: int = 0

        self.s: int = -1
        self.t: int = 0

    def load_photo(self, data: np.ndarray):
        self.photo_height = data.shape[0]
        self.photo_width = data.shape[1]

        self.nodes_count = self.photo_width * self.photo_height

        self.nodes = list(range(self.nodes_count))

        self.adj_list: List[List[Arc]] = [[] for i in range(self.nodes_count)]
        self.residual_adj_list = [[] for i in range(self.nodes_count)]

        for i in range(self.nodes_count):
            for j in range(self.nodes_count):
                u: int = self.nodes[i]
                v: int = self.nodes[j]

                arc = Arc(u, v)
                if abs(u - v) == self.photo_width or abs(u - v) == 1 or u == self.s or v == self.t:

                    arc.capacity = f(data[u // self.photo_width][u % self.photo_width],
                                     data[v // self.photo_width][v % self.photo_width])
                else:
                    arc.capacity = 0

                r_arc = Arc(v, u)
                r_arc.is_residual = True
                arc.residual = r_arc

                self.adj_list[u].append(arc)
                self.residual_adj_list[v].append(r_arc)

    def set_background(self, pos: Tuple[int]):
        self.s = pos[0] * self.photo_width + pos[1]

    def set_foreground(self, pos: Tuple[int]):
        self.t = pos[0] * self.photo_width + pos[1]

    def shortest_augmentative_path(self):
        nodes: List[int] = [self.s]
        visited: List[int] = [None] * self.nodes_count
        while nodes != [] and nodes[0] != self.t:

            for arc in self.adj_list[nodes[0]] + self.residual_adj_list[nodes[0]]:
                if arc.capacity - arc.flow > 0 and visited[arc.v] is None:
                    visited[arc.v] = arc
                    nodes.append(arc.v)
            nodes.pop(0)

        if nodes:
            path: List[Arc] = []
            node: int = self.t
            while node != self.s:
                path.append(visited[node])
                node = visited[node].u
            path.reverse()
            return path

        return None

    @staticmethod
    def augment(path: List[Arc]):
        flow: float = float("inf")
        for arc in path:
            if arc.capacity - arc.flow < flow:
                flow = arc.capacity - arc.flow

        for arc in path:
            arc.flow += flow
            if not arc.is_residual:
                arc.residual.capacity = arc.flow

    def edmonds_karp(self):
        while True:
            path: List[Arc] = self.shortest_augmentative_path()
            if path is None:
                break
            else:
                self.augment(path)

    def find_cut(self):
        nodes: List[int] = [self.s]
        S: List[int] = []
        visited: List[int] = [None] * self.nodes_count
        visited[self.s] = True
        while nodes:

            for arc in self.adj_list[nodes[0]] + self.residual_adj_list[nodes[0]]:
                if arc.capacity - arc.flow > 0 and visited[arc.v] is None:
                    visited[arc.v] = arc
                    nodes.append(arc.v)

            S.append(nodes.pop(0))
        T: List[Arc] = []
        for node in self.nodes:
            if visited[node] is None:
                T.append(node)

        T = S
        for i in range(len(T)):
            T[i] = [T[i] // self.photo_width, T[i] % self.photo_width]
        return T


@timeit
def cut(photo: np.ndarray, foreground: Tuple[int], background: Tuple[int], shape: Tuple[int] = (20, 20)):
    real_shape = photo.shape
    photo = cv2.resize(photo, (20, 20))

    foreground = (int(foreground[0] * shape[0] / real_shape[0]), int(foreground[1] * shape[1] / real_shape[1]))
    background = (int(background[0] * shape[0] / real_shape[0]), int(background[1] * shape[1] / real_shape[1]))
    n = Network()

    n.set_foreground(foreground)
    n.set_background(background)
    n.load_photo(photo)

    n.edmonds_karp()

    T = n.find_cut()
    new_photo = np.copy(photo)

    for (x, y) in T:
        new_photo[x][y] = np.zeros(3)

    new_photo = cv2.resize(new_photo, real_shape[0:2])
    return new_photo

