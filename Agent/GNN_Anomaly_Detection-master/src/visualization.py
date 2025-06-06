import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

graph_index = 0
embedding_index =0

def visualize_directed_graph(self):
    global graph_index
    plt.clf()

    # Get the nodes for each subset
    client_ip_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]["side"] == "Client-IP"]
    client_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]["side"] == "Client"]
    # flow_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]["side"] == "Flow"]
    server_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]["side"] == "Server"]
    
    # Initialize positions dictionary
    pos = {}
    
    # Set positions for Client-IP nodes
    for i, node in enumerate(client_ip_nodes):
        pos[node] = (-2, i * 2.0 / len(client_ip_nodes))

    # Set positions for Client nodes
    for i, node in enumerate(client_nodes):
        pos[node] = (-1, i * 2.0 / len(client_nodes))

    # # Set positions for Flow nodes
    # for i, node in enumerate(flow_nodes):
    #     pos[node] = (0, i * 2.0 / len(flow_nodes))

    # Set positions for Server nodes
    for i, node in enumerate(server_nodes):
        pos[node] = (1, i * 2.0 / len(server_nodes))
    
    # Draw nodes
    node_colors = [self.graph.nodes[node]["color"] for node in self.graph.nodes]
    nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=400, node_shape='o')
    
    # Draw node labels
    node_labels = {node: node for node in self.graph.nodes}
    nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_color="white", font_size=6, verticalalignment='center')
    
    # Draw edges
    nx.draw_networkx_edges(self.graph, pos, edge_color='gray', node_size=700)
    
    # plt.savefig(f'../output/imgs_dos_clusters/graph_imgs/graph{graph_index}.png')
    plt.ion()
    plt.show()
    plt.pause(0.1)
    graph_index += 1

# Plot the graph embeddings
def plot_embeddings(embeddings, graph: nx.graph):
    global embedding_index
    # Convert embeddings to a NumPy array
    perplexity = 5
    embeddings_array = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings.detach().cpu().numpy())
    
    # Prepare the plot
    plt.clf()
    ids = list(graph.nodes)
    
    # Create a dictionary to map colors to labels
    color_label_map = {}
    for id in ids:
        if graph.nodes[id]["side"] == "Client":
            color = graph.nodes[id]["color"]
            label = graph.nodes[id]["ip"].split(":")[0]
            if color not in color_label_map:
                color_label_map[color] = label
    
    # Plot embeddings according to their anomaly score
    for i, embedding in enumerate(embeddings_array):
        id = ids[i]
        if graph.nodes[id]["side"] != "Client":
            continue
        plt.scatter(embedding[0], embedding[1], c=graph.nodes[id]["color"], alpha=0.5, label=graph.nodes[id]["ip"].split(":")[0])

    # Create the legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in color_label_map]
    labels = [label for label in color_label_map.values()]
    # plt.legend(handles, labels, title="Legend", loc='best')
    plt.legend(handles, labels, loc='best')
    
    # Finalize the plot
    plt.title("Graph Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(f'../output/imgs_dos_clusters/embedding_imgs/embedding{embedding_index}.png')
    embedding_index += 1
    
# Plot the ann indexes
def plot_ann_indexes(vectors):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced_vectors = tsne.fit_transform(vectors)

    # plt.figure(figsize=(8, 6))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', marker='o')
            
    plt.title('t-SNE Projection of ANN Vectors')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.ion()
    plt.show()
    plt.pause(0.1)