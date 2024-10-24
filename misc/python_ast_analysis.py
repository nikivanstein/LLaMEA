# Uses Python Abstract Syntax Trees to extract graph characteristics
import ast
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import argparse

#Parse Python AST and build a graph
class BuildAST(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_node = 0
        self.node_stack = []

    def generic_visit(self, node):
        node_id = self.current_node
        self.graph.add_node(node_id, label=type(node).__name__)
        
        if self.node_stack:
            parent_id = self.node_stack[-1]
            self.graph.add_edge(parent_id, node_id)
        
        self.node_stack.append(node_id)
        self.current_node += 1
        
        super().generic_visit(node)
        
        self.node_stack.pop()
    
    def build_graph(self, root):
        self.visit(root)
        return self.graph

# Function to extract graph characteristics
def analyze_graph(G):
    depths = dict(nx.single_source_shortest_path_length(G, min(G.nodes())))
    degrees = sorted((d for n, d in G.degree()), reverse=True)
    leaf_depths = [depth for node, depth in depths.items() if G.out_degree(node) == 0] #depth from root to leaves
    clustering_coefficients = list(nx.clustering(G).values())
    #Additional Features (not in paper)
    #Convert the directed graph to an undirected graph to avoid SCC problems
    undirected_G = G.to_undirected()
    if nx.is_connected(undirected_G): #check if undirected graph is connected
        diameter = nx.diameter(undirected_G)
        radius = nx.radius(undirected_G)
        avg_shortest_path = nx.average_shortest_path_length(undirected_G)
        avg_eccentricity = np.mean(list(nx.eccentricity(undirected_G).values()))
    else:
        #Calculate diameter of the largest strongly connected component
        largest_cc = max(nx.connected_components(undirected_G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        radius = nx.radius(subgraph)
        avg_shortest_path = nx.average_shortest_path_length(subgraph)
        avg_eccentricity = np.mean(list(nx.eccentricity(subgraph).values()))
    edge_density = G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)) if G.number_of_nodes() > 1 else 0

    return {
        #Number of Nodes and Edges
        "Nodes": G.number_of_nodes(),
        "Edges":  G.number_of_edges(),
        #Degree Analysis
        "Degrees": degrees,
        "Max Degree": max(degrees),
        "Min Degree": min(degrees),
        "Mean Degree": np.mean(degrees),
        "Degree Variance": np.var(degrees),
        #Transitivity
        "Transitivity":  nx.transitivity(G),
        #Depth analysis
        "Depths": leaf_depths,
        "Max Depth": max(leaf_depths),
        "Min Depth": min(leaf_depths),
        "Mean Depth": np.mean(leaf_depths),
        #Clustering Coefficients
        "Clustering Coefficients": clustering_coefficients,
        "Max Clustering": max(clustering_coefficients),
        "Min Clustering": min(clustering_coefficients),
        "Mean Clustering": nx.average_clustering(G),
        "Clustering Variance": np.var(clustering_coefficients),
        #Entropy
        "Degree Entropy": entropy(degrees),
        "Depth Entropy": entropy(leaf_depths),
        #Additional features (not in paper)
        "Betweenness Centrality": nx.betweenness_centrality(G),
        "Eigenvector Centrality": nx.eigenvector_centrality_numpy(G, max_iter=500),
        "Assortativity": nx.degree_assortativity_coefficient(G),
        "Average Eccentricity": avg_eccentricity,
        "Diameter": diameter,
        "Radius": radius,
        "Pagerank": nx.pagerank(G, max_iter=500),
        "Edge Density": edge_density,
        "Average Shortest Path": avg_shortest_path,
    }

def visualize_graph(G):
    pos = graphviz_layout(G, prog='dot')
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color="lightblue", 
            font_size=8, font_weight="bold", arrows=True)
    plt.savefig("graph1.pdf")

#Function to create graph out of AST
def process_file(path, visualize):
    with open(path, 'r') as file:
        python_code = file.read()
    root = ast.parse(python_code)
    build = BuildAST()
    G = build.build_graph(root)
    stats = analyze_graph(G)
    if (visualize==True): #visualize graph
        visualize_graph(G)
    return stats

def aggregate_stats(results):
    print("Aggregate Statistics:")
    print("Total Nodes:", sum(result['Nodes'] for result in results))
    print("Total Edges:", sum(result['Edges'] for result in results))
    print("Average Transitivity:", sum(result['Transitivity'] for result in results) / len(results))
    print("Max Depth:", max(result['Max Depth'] for result in results))
    print("Average Degree Mean:", np.mean([result['Mean Degree'] for result in results]))
    print("Average Clustering Coefficient:", np.mean([result['Mean Clustering'] for result in results]))
    print("Average Eccentricity:", np.mean([result['Average Eccentricity'] for result in results]))
    print("Average Edge Density:", np.mean([result['Edge Density'] for result in results]))

def print_results(stats, file):
    print("Statistics for file:", file)
    print("Number of nodes:", stats['Nodes'])
    print("Number of Edges:", stats['Edges'])
    print("Degrees:", stats['Degrees'])
    print("Maximum Degree:", stats['Max Degree'])
    print("Minimum Degree:", stats['Min Degree'])
    print("Mean Degree:", stats['Mean Degree'])
    print("Degree Variance:", stats['Degree Variance'])
    print("Transitivity:", stats['Transitivity'])
    print("Leaf Depths:", stats['Depths'])
    print("Max Depth:", stats['Max Depth'])
    print("Min Depth:", stats['Min Depth'])
    print("Mean Depth:", stats['Mean Depth'])
    print("Clustering Coefficients:", stats['Clustering Coefficients'])
    print("Max Clustering:", stats['Max Clustering'])
    print("Min Clustering:", stats['Min Clustering'])
    print("Mean Clustering:", stats['Mean Clustering'])
    print("Clustering Variance:", stats['Clustering Variance'])
    print("Degree Entropy:", stats['Degree Entropy'])
    print("Depth Entropy:", stats['Depth Entropy'])
    print("Betweenness Centrality:", stats['Betweenness Centrality'])
    print("Eigenvector Centrality:", stats['Eigenvector Centrality'])
    print("Assortativity:", stats['Assortativity'])
    print("Average Eccentricity:", stats['Average Eccentricity'])
    print("Diameter:", stats['Diameter'])
    print("Radius:", stats['Radius'])
    print("Pagerank:", stats['Pagerank'])
    print("Edge Density:", stats['Edge Density'])
    print("Average Shortest Path:", stats['Average Shortest Path'])
    print("")

def main(file_paths, visualize, filename="graph.pdf"):
    results = []
    for file_path in file_paths:
        stats = process_file(file_path, visualize, filename)
        results.append(stats)
        print_results(stats, file_path)
    #aggregate_stats(results)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        file_paths = sys.argv[1:] #take multiple files
        parser = argparse.ArgumentParser(description="Analyze Clang ASTs and extract graph features.")
        #The -g flag should be specific before input C files to visualize graphs
        parser.add_argument("-g", "--graph", action="store_true", help="Visualize the graph after processing.")
        parser.add_argument("files", nargs='+', help="List of input Clang AST files to process.")
        args = parser.parse_args()
        main(file_paths=args.files, visualize=args.graph)
    else:
        print("Error: Specify Python file path(s).")    
