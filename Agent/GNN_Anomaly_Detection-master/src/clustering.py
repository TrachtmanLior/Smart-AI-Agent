import numpy as np

# For clustering
import hdbscan

# For anomalies
from sklearn.metrics import pairwise_distances_argmin_min

from config import clustering_threshold, dataset_type

# Function to perform clustering algorithm
def clustering_algorithm(embeddings): 
    
    # Initialize HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True, metric='euclidean')
    
    # Perform clustering
    cluster_labels = clusterer.fit_predict(embeddings)
    
    return cluster_labels

#Function to calculte the amount of every cluster
def count_vectors_in_clusters(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique_labels, counts))
    return cluster_counts

# Function to measure distance of a cluster
def find_centroids(vectors, labels):
    # Calculate centroids for each cluster
    centroids = []
    for cluster in np.unique(labels):
        cluster_points = vectors[labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    
    # Convert centroids to a NumPy array
    return np.array(centroids)

def calculate_centroid_distances(centroids):
    distances = []
    for i in range(len(centroids)):
        list_without_current = np.delete(centroids, i, axis=0)
        current_distance = pairwise_distances_argmin_min(centroids, list_without_current, metric='euclidean')[1]
        distances.append(current_distance[i])
    return distances

# Function to measure density of a cluster
def calculate_density(vectors, labels):
    densities = []
    for cluster in np.unique(labels):
        cluster_points = vectors[labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        distances = pairwise_distances_argmin_min(cluster_points, [centroid])[1]
        current_density = np.mean(distances)
        densities.append(current_density)
    return densities


def check_all_anomalies(graph, embeddings, clusters, pred, node_to_index, to_print=True):
    list_nodes = list(graph.nodes)
    
    def check_and_print_anomalies(elements, description = None):
        # calculate anomalies
        elements = elements[1:] #ignore the isolated nodes
        avg_elements = np.mean(elements)
        std_elements = np.std(elements)
        
        unusual_elements = [i for i in range(len(elements)) 
                            if (elements[i] > avg_elements + clustering_threshold * std_elements) 
                            or  (elements[i] < avg_elements - clustering_threshold * std_elements)]
        
        # print anomalies & update predicted label
        for cluster in unusual_elements:
            if to_print:
                print(f"Cluster {cluster} is anomaly with the nodes:")
            for i, node in enumerate(clusters):
                if node == cluster:
                    curr_node = graph.nodes[list_nodes[i]]
                    curr_node["pred"] = True
                    curr_node["cluster_pred"] = True
                    curr_node["color"] = "lightgreen" if curr_node["label"] else "yellow"
                    curr_node["cluster"] = cluster # CHECK
                    if dataset_type != 'packets_csv':
                        pred[node_to_index[list_nodes[i]]] = True
                    
                    if to_print:
                        print(f"found ({description}) anomaly in node: {graph.nodes[list_nodes[i]]}")
            if to_print:
                print()
        
    # Check for anomaly clusters amount
    cluster_counts = count_vectors_in_clusters(clusters)
    check_and_print_anomalies(list(cluster_counts.values()), 'amount')

    # Check for anomaly clusters densities
    cluster_densities = calculate_density(embeddings, clusters)
    check_and_print_anomalies(cluster_densities, "density")
    
    # Check for anomaly distances between centroids
    centroids = find_centroids(embeddings, clusters)
    centroid_distances = calculate_centroid_distances(centroids)
    check_and_print_anomalies(centroid_distances, "distances")
    if to_print:
        print()
