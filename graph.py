import math
from typing import List, Tuple

import cv2
import numpy as np
from timing import timeit


class Arc:
    def __init__(self, u: int, v: int, uf: int, residual=None):
        self.u = u
        self.v = v
        self.uf = uf  # flow upper bound
        self.f = 0
        self.residual = residual
        self.is_main = True

    def __repr__(self):
        return f"u: {self.u} \t v: {self.v} \t r: {self.r}"


class Image:
    def __init__(self, data: np.ndarray):
        self.height: int = data.shape[0]
        self.width: int = data.shape[1]
        self.n: int = self.width * self.height
        self.data: np.ndarray = data

        self.sigma: int = 30

        self.s = self.n
        self.t = self.n + 1
        self.n += 2

        self.adj_list = None
        self.residual_adj_list = None

    def pixel2vertice(self, x: int, y: int):
        return self.width * x + y

    def vertice2pixel(self, n: int):
        return n // self.width, n % self.width

    def get_neighbours(self, n: int):
        neighbours = []
        if n >= self.width:
            neighbours.append(n - self.width)
        if n < self.width * (self.height - 1):
            neighbours.append(n + self.width)
        if n % self.width > 0:
            neighbours.append(n - 1)
        if n % self.width < self.width - 1:
            neighbours.append(n + 1)

        return neighbours

    def nlink_arc(self, u: int, v: int):
        pixel_u = self.vertice2pixel(u)
        pixel_v = self.vertice2pixel(v)

        data_u = np.average(self.data[pixel_u])
        data_v = np.average(self.data[pixel_v])

        w: int = int(100 * np.exp(-(data_u - data_v) ** 2 / (2 * self.sigma ** 2)))
        arc = Arc(u, v, w)

        return arc

    @timeit
    def create_graph(self):
        self.adj_list = [[] for i in range(self.n)]

        for u in range(self.n - 2):
            for v in self.get_neighbours(u):
                arc = self.nlink_arc(u, v)
                self.adj_list[u].append(arc)

    def add_foreground(self, pixels: List[Tuple[int]]):
        max_arc_uf = self.adj_list[0][0].uf
        for row in self.adj_list:
            for arc in row:
                if max_arc_uf < arc.uf:
                    max_arc_uf = arc.uf
        max_arc_uf *= 10
        for pixel in pixels:
            u = self.pixel2vertice(*pixel)
            arc = Arc(self.s, u, max_arc_uf)
            self.adj_list[self.s].append(arc)

    def add_background(self, pixels: List[Tuple[int]]):
        max_arc_uf = self.adj_list[0][0].uf
        for row in self.adj_list:
            for arc in row:
                if max_arc_uf < arc.uf:
                    max_arc_uf = arc.uf
        max_arc_uf *= 10
        for pixel in pixels:
            u = self.pixel2vertice(*pixel)
            arc = Arc(u, self.t, max_arc_uf)
            self.adj_list[u].append(arc)

    def create_residual_graph(self):
        self.residual_adj_list = [[] for i in range(self.n)]
        for u in range(self.n):
            for arc in self.adj_list[u]:
                self.residual_adj_list[u].append(arc)


                r_arc = Arc(arc.v, arc.u, 0)
                r_arc.is_main = False

                arc.residual = r_arc
                r_arc.residual = arc

                self.residual_adj_list[arc.v].append(r_arc)

    def do_bfs(self, s, t):
        queue = [s]
        pred = [None] * self.n
        pred[s] = s

        is_found = False

        while queue and not is_found:
            u = queue.pop(0)
            if u == t:
                is_found = True
                continue

            for arc in self.residual_adj_list[u]:
                if not pred[arc.v] and arc.uf - arc.f > 0:
                    queue.append(arc.v)
                    pred[arc.v] = arc

        return pred

    def get_shortest_augmentative_path(self):
        pred = self.do_bfs(self.s, self.t)
        if pred[self.t] is None:
            return None


        augmentative_path = []
        u = self.t
        while u != self.s:
            augmentative_path.append(pred[u])
            u = pred[u].u

        augmentative_path.reverse()
        return augmentative_path

    def augment_flow(self, augmentative_path: List[Arc]):
        delta = float("inf")
        for arc in augmentative_path:
            if arc.uf - arc.f < delta:
                delta = arc.uf - arc.f

        for arc in augmentative_path:
            arc.f += delta
            if arc.residual:
                arc.residual.uf += delta


    def edmond_karp(self):
        while True:
            shortest_augmentative_path = self.get_shortest_augmentative_path()
            if not shortest_augmentative_path:
                return
            self.augment_flow(shortest_augmentative_path)

    @timeit
    def do_cut(self, foreground_pixels: List[Tuple[int]], background_pixels: List[Tuple[int]]):
        self.create_graph()
        self.add_background(background_pixels)
        self.add_foreground(foreground_pixels)
        self.create_residual_graph()
        self.edmond_karp()

    def get_background(self):
        pred = self.do_bfs(self.s, self.t)
        background = []
        for u in range(self.n):
            if not pred[u]:
                background.append(u)
        background.remove(self.t)

        for i in range(len(background)):
            background[i] = self.vertice2pixel(background[i])


        return background

    def get_foreground(self):
        pred = self.do_bfs(self.s, self.t)
        foreground = []
        for u in range(self.n):
            if pred[u]:
                foreground.append(u)
        foreground.remove(self.s)
        for i in range(len(foreground)):
            foreground[i] = self.vertice2pixel(foreground[i])

        return foreground

