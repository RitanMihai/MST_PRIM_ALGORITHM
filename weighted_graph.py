import heapq
import random
from enum import Enum
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx import dijkstra_path

from fibonacci_heap import FibonacciHeap


class WeightedGraph:
    class GRAPH_DENSITY(Enum):
        N_MIN_ONE = "N-1"
        SPARSE = "SPARSE"
        MEDIUM = "MEDIUM"
        COMPLETE = "COMPLETE"

    class GRAPH_TYPE(Enum):
        CLASSIC = "CLASSIC"
        MAZE = "MAZE" # Used to generate images of a maze

    def __init__(self, num_nodes: int = None, num_edges: int = None, density: GRAPH_DENSITY = None,
                 width: int = None, height: int = None):
        if num_nodes is not None:
            if num_edges is not None:
                self._init_with_num_edges(num_nodes, num_edges)
            elif density is not None:
                self._init_with_density(num_nodes, density)
            else:
                raise ValueError("Either num_edges or density must be provided for classic graph creation")
        elif width is not None and height is not None:
            self.width = width
            self.height = height
            self._init_maze(width, height)
        else:
            raise ValueError("Invalid arguments for graph creation")

    def _init_with_num_edges(self, num_nodes: int, num_edges: int):
        self.graph = nx.gnm_random_graph(num_nodes, num_edges)
        for (u, v) in self.graph.edges():
            self.graph[u][v]['weight'] = random.randint(0, 10)

    def _init_with_density(self, num_nodes: int, density: GRAPH_DENSITY):
        match density:
            case WeightedGraph.GRAPH_DENSITY.N_MIN_ONE:
                self.graph = nx.gnm_random_graph(num_nodes, num_nodes - 1)
            case WeightedGraph.GRAPH_DENSITY.SPARSE:
                self.graph = nx.gnm_random_graph(num_nodes, max(1, num_nodes * (num_nodes // 8)))
            case WeightedGraph.GRAPH_DENSITY.MEDIUM:
                self.graph = nx.gnm_random_graph(num_nodes, num_nodes * (num_nodes // 4))
            case WeightedGraph.GRAPH_DENSITY.COMPLETE:
                self.graph = nx.gnm_random_graph(num_nodes, num_nodes * (num_nodes - 1) // 2)
            case _:  # DEFAULT COMPLETE
                self.graph = nx.gnm_random_graph(num_nodes, num_nodes * (num_nodes - 1) // 2)

        for (u, v) in self.graph.edges():
            self.graph[u][v]['weight'] = random.randint(0, 10)

    def _init_maze(self, width: int, height: int):
        self.graph = nx.grid_2d_graph(width, height)
        for (u, v) in self.graph.edges():
            self.graph[u][v]['weight'] = random.randint(1, 10)

    def prim_lazy_adjacency_matrix(self):
        nodes = list(self.graph.nodes)
        n = len(nodes)
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=nodes)

        MST = []
        selected = [False] * n
        selected[0] = True

        for _ in range(n - 1):
            min_weight = float('inf')
            u = -1
            v = -1

            for i in range(n):
                if selected[i]:
                    for j in range(n):
                        if not selected[j] and adj_matrix[i][j] != 0:
                            if adj_matrix[i][j] < min_weight:
                                min_weight = adj_matrix[i][j]
                                u = i
                                v = j

            if u != -1 and v != -1:
                MST.append((nodes[u], nodes[v], min_weight))
                selected[v] = True

        return MST

    def prim_binary_heap_adjacency_list(self, start):
        MST = []
        visited = set([start])
        edges = [(data['weight'], start, to) for to, data in self.graph[start].items()]
        heapq.heapify(edges)

        # while len(visited) != len(self.graph.edges):
        while edges:
            weight, frm, to = heapq.heappop(edges)
            if to not in visited:
                visited.add(to)
                MST.append((frm, to, weight))
                for next_to, data in self.graph[to].items():
                    if next_to not in visited:
                        heapq.heappush(edges, (data['weight'], to, next_to))

        return MST

    def prim_fibonacci_heap_adjacency_list(self, start):
        MST = []
        visited = set()
        fib_heap = FibonacciHeap()
        nodes = {vertex: fib_heap.insert(float('inf'), vertex) for vertex in self.graph}
        fib_heap.decrease_key(nodes[start], 0)

        while fib_heap.total_nodes:
            min_node = fib_heap.extract_min()
            vertex = min_node.value
            visited.add(vertex)
            if vertex != start:
                for to, data in self.graph[vertex].items():
                    if to in visited and data['weight'] == min_node.key:
                        MST.append((to, vertex, data['weight']))
                        break

            for to, data in self.graph[vertex].items():
                if to not in visited and data['weight'] < nodes[to].key:
                    fib_heap.decrease_key(nodes[to], data['weight'])

        return MST

    def dijkstra_path(self, start, goal):
        return nx.shortest_path(self.graph, source=start, target=goal, weight='weight', method='dijkstra')

    # PLOTTING
    def plot_graph(self, mst=None):
        pos = nx.spring_layout(self.graph)  # positions for all nodes
        plt.figure(figsize=(12, 8))

        # Draw the original graph in light gray
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightgray', font_weight='bold',
                font_color='black')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        if mst is not None:
            # Highlight the MST edges in red
            mst_edges = [(u, v) for u, v, _ in mst]
            nx.draw_networkx_edges(self.graph, pos, edgelist=mst_edges, edge_color='r', width=2)

        plt.title("Graph and its Minimum Spanning Tree (MST)")
        plt.show()

    # Below methods are just for fun
    def plot_labyrinth(self):
        mst = nx.minimum_spanning_tree(self.graph, algorithm="prim")

        maze = np.zeros((2 * self.height + 1, 2 * self.width + 1), dtype=int)
        maze[1::2, 1::2] = 1  # Initialize cells

        for (u, v) in mst.edges():
            ux, uy = u
            vx, vy = v
            maze[2 * uy + 1, 2 * ux + 1] = 1
            maze[2 * vy + 1, 2 * vx + 1] = 1
            # Path between cells
            maze[2 * uy + 1 + (vy - uy), 2 * ux + 1 + (vx - ux)] = 1

        plt.figure(figsize=(10, 10))
        plt.imshow(maze, cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.show()

    def visualize_labyrinth_with_path(self):
        mst = nx.minimum_spanning_tree(self.graph, algorithm="prim")

        maze = np.zeros((2 * self.height + 1, 2 * self.width + 1), dtype=int)
        maze[1::2, 1::2] = 1  # Initialize cells

        for (u, v) in mst.edges():
            ux, uy = u
            vx, vy = v
            maze[2 * uy + 1, 2 * ux + 1] = 1
            maze[2 * vy + 1, 2 * vx + 1] = 1
            # Path between cells
            maze[2 * uy + 1 + (vy - uy), 2 * ux + 1 + (vx - ux)] = 1

        start, goal = (0, 0), (self.width - 1, self.height - 1)
        shortest_path = dijkstra_path(mst, start, goal)

        # Highlight the shortest path
        for i in range(len(shortest_path) - 1):
            ux, uy = shortest_path[i]
            vx, vy = shortest_path[i + 1]
            maze[2 * uy + 1, 2 * ux + 1] = 2
            maze[2 * vy + 1, 2 * vx + 1] = 2
            # Path between cells
            maze[2 * uy + 1 + (vy - uy), 2 * ux + 1 + (vx - ux)] = 2

        plt.figure(figsize=(10, 10))
        plt.imshow(maze, cmap='summer', interpolation='nearest')
        plt.axis('off')
        plt.show()

    def prim_binary_heap_adjacency_list_with_plot(self, start):
        MST = []
        visited = set([start])
        edges = [(data['weight'], start, to) for to, data in self.graph[start].items()]
        heapq.heapify(edges)

        pos = nx.spring_layout(self.graph)  # Positions for all nodes

        plt.figure(figsize=(14, 8))
        self._draw_graph(pos, MST, start, visited)
        self._draw_priority_queue(edges)
        plt.show(block=False)
        plt.pause(1)  # Pause to update the plot

        while edges:
            weight, frm, to = heapq.heappop(edges)
            if to not in visited:
                visited.add(to)
                MST.append((frm, to, weight))
                for next_to, data in self.graph[to].items():
                    if next_to not in visited:
                        heapq.heappush(edges, (data['weight'], to, next_to))

                # Update the plot with the current MST and priority queue
                plt.clf()
                self._draw_graph(pos, MST, to, visited)
                self._draw_priority_queue(edges)
                plt.show(block=False)
                plt.pause(1)  # Pause to update the plot

        plt.show()
        return MST

    def _draw_graph(self, pos, MST, current_node, visited):
        node_colors = []
        for node in self.graph.nodes():
            if node == current_node:
                node_colors.append('orange')
            elif node in visited:
                node_colors.append('gray')
            else:
                node_colors.append('aqua')

        edge_colors = []
        for u, v in self.graph.edges():
            if (u, v) in [(x, y) for x, y, _ in MST] or (v, u) in [(x, y) for x, y, _ in MST]:
                edge_colors.append('red')
            else:
                edge_colors.append('black')

        plt.subplot(121)
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color=node_colors, font_weight='bold',
                font_color='black', edge_color=edge_colors)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Graph with MST construction")

    def _draw_priority_queue(self, edges):
        plt.subplot(122)
        plt.title("Priority Queue")
        plt.xlabel("Edge")
        plt.ylabel("Weight")
        queue_data = sorted(edges)
        y_labels = [f"{frm}-{to}" for weight, frm, to in queue_data]
        weights = [weight for weight, frm, to in queue_data]
        plt.barh(y_labels, weights, color='blue')
        plt.gca().invert_yaxis()  # Highest priority at the top
        plt.grid(True)
