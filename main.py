import json
import os
import sys
import time

import networkx as nx
from matplotlib import pyplot as plt

from weighted_graph import WeightedGraph


def measure_execution_time(algorithm, start=None):
    start_time = time.perf_counter()
    if start is not None:
        algorithm(start)
    else:
        algorithm()
    end_time = time.perf_counter()
    return end_time - start_time


def run_experiments(runs=10):
    num_nodes_list = [10, 20, 50, 100, 200, 500, 800]
    results = []

    for density in WeightedGraph.GRAPH_DENSITY:
        for num_nodes in num_nodes_list:
            print(f"Running experiments for {num_nodes} nodes on {density} graph")

            lazy_times = []
            binary_times = []
            fibonacci_times = []
            networkx_prim = []

            for _ in range(runs):
                wg = WeightedGraph(num_nodes=num_nodes, density=density)

                start_node = list(wg.graph.nodes)[0]
                lazy_times.append(measure_execution_time(wg.prim_lazy_adjacency_matrix))
                binary_times.append(measure_execution_time(wg.prim_binary_heap_adjacency_list, start_node))
                fibonacci_times.append(measure_execution_time(wg.prim_fibonacci_heap_adjacency_list, start_node))

                # NetworkX default Prim
                start_time = time.perf_counter()
                nx.minimum_spanning_tree(wg.graph, "prim")
                end_time = time.perf_counter()
                networkx_prim.append(end_time - start_time)

            results.append({
                "density": density.name,
                "num_nodes": num_nodes,
                "num_edges": len(wg.graph.edges),
                "lazy_avg_time": sum(lazy_times) / runs,
                "binary_avg_time": sum(binary_times) / runs,
                "fibonacci_avg_time": sum(fibonacci_times) / runs,
                "networkx_avg_time": sum(networkx_prim) / runs
            })

    with open('mst_results_backup.json', 'w') as f:
        json.dump(results, f, indent=4)


def plot_results(selected_methods):
    with open('mst_results_backup.json', 'r') as f:
        results = json.load(f)

    densities = list(WeightedGraph.GRAPH_DENSITY)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    density_map = {
        'N_MIN_ONE': 0,
        'SPARSE': 1,
        'MEDIUM': 2,
        'COMPLETE': 3
    }

    density_results = {density.name: [] for density in densities}
    for result in results:
        density_results[result["density"]].append(result)

    selected_methods = selected_methods.split(';')
    all_methods = ["lazy_avg_time", "binary_avg_time", "fibonacci_avg_time", "networkx_avg_time"]
    method_labels = {
        "lazy_avg_time": "Lazy Adjacency Matrix",
        "binary_avg_time": "Binary Heap Adjacency List",
        "fibonacci_avg_time": "Fibonacci Heap Adjacency List",
        "networkx_avg_time" : "NetworkX Prim"
    }

    if '*' in selected_methods:
        methods_to_plot = all_methods
    else:
        methods_to_plot = [method for method in all_methods if method in selected_methods]

    for density, index in density_map.items():
        ax = axs[index // 2, index % 2]
        density_result = density_results[density]

        num_nodes = sorted(set(res["num_nodes"] for res in density_result))

        for method in methods_to_plot:
            times = [next(res[method] for res in density_result if res["num_nodes"] == n) for n in num_nodes]
            ax.plot(num_nodes, times, label=method_labels[method])

        ax.set_title(density)
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Average Execution Time (seconds)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def main_menu():
    print("Select an option:")
    print("1. Create and visualize labyrinth")
    print("2. Create and visualize labyrinth with shortest path")
    print("3. Create and visualize graph and MST")
    print("4. Run performance tests")
    print("5. Plot performance tests")
    print("6. Plot the steps of the Prim Alg")
    print("7. Exit")

    choice = input("Enter your choice: ")
    return choice

def create_graph_menu():
    nodes = int(input("Enter number of nodes: "))

    while True:
        density_type = input(
            "Select your density type:\n1. Insert exact number of edges\n2. Select a DENSITY TYPE\nEnter choice: ")
        if density_type == '1':
            edges = int(input("Enter number of edges: "))
            graph = WeightedGraph(num_nodes=nodes, num_edges=edges)
            return graph
        elif density_type == '2':
            density_type = input(
                "Select type of density:\n1. N-1\n2. SPARSE\n3. MEDIUM\n4. COMPLETE\nEnter choice: ")
            density = None
            match density_type:
                case "1" | "N-1":
                    density = WeightedGraph.GRAPH_DENSITY.N_MIN_ONE
                case "2" | "SPARSE":
                    density = WeightedGraph.GRAPH_DENSITY.SPARSE
                case "3" | "MEDIUM":
                    density = WeightedGraph.GRAPH_DENSITY.MEDIUM
                case "4" | "COMPLETE":
                    density = WeightedGraph.GRAPH_DENSITY.COMPLETE
                case _:
                    print("Invalid choice. Please try again.")
                    continue
            graph = WeightedGraph(num_nodes=nodes, density=density)
            return graph
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    while True:
        choice = main_menu()
        match choice:
            case '1':
                width, height = int(input("Width: ")), int(input("Height: "))
                # Transmit as tuple
                graph = WeightedGraph(width=width, height=height)
                graph.plot_labyrinth()
            case '2':
                width, height = int(input("Width: ")), int(input("Height: "))
                graph = WeightedGraph(width=width, height=height)
                graph.visualize_labyrinth_with_path()
            case '3':
                graph = create_graph_menu()
                mst = graph.prim_binary_heap_adjacency_list(0)
                graph.plot_graph(mst)
            case '4':
                run_experiments()
            case '5':
                if os.path.exists('mst_results_backup.json'):
                    print("Available methods to plot: 'lazy_avg_time', 'binary_avg_time', 'fibonacci_avg_time', 'networkx_avg_time' "
                          "\nex: binary_avg_time;fibonacci_avg_time")
                    selected_methods = input("Enter methods to plot separated by `;` or `*` for all: ")
                    plot_results(selected_methods)
                else:
                    print("First please run tests.")
            case '6':
                graph = create_graph_menu()
                start = list(graph.graph.nodes)[0]
                graph.prim_binary_heap_adjacency_list_with_plot(start)
            case '7':
                sys.exit()
