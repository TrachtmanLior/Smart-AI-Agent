import numpy as np
from annoy import AnnoyIndex
from datetime import datetime

from config import features, ann_threshold, ann_history_threshold, anomaly_score_history_size, output_size

def node_to_str(node) -> str:
    print_str = ''
    for feature in features:
        feature_value = node[feature] / node['flows']
        print_str += f'{feature}: {feature_value} '

    side = node['side']

    if side == 'Client':
        ip = node['ip']
        print_str += f'\n on Client ip: {ip} \n'
    
    elif side == 'Server':
        ip = node['ip']
        print_str += f'\n on Server ip: {ip} \n'

    elif side == 'Flow':
        stream_number = node['stream_number']
        sip = node['sip']
        dip = node['dip']
        
        print_str += f'\n on Flow number: {stream_number} with Client ip: {sip} and Server ip: {dip} \n'
    
    return print_str

def print_anomalies(graph, anomaly_node_id, description):
    anomaly_node = graph.nodes[anomaly_node_id]
    # anomaly_node_str = node_to_str(anomaly_node)
    # ts = datetime.fromtimestamp(anomaly_node["packet_index"]).strftime('%Y-%m-%d %H:%M:%S')
    # ts = datetime.fromtimestamp(int(anomaly_node["packet_index"])).strftime('%Y-%m-%d %H:%M:%S')
    # print(f'found ({description}) anomaly on packet number {ts} (node id: {anomaly_node_id}): {anomaly_node_str}')
    print(f'found anomaly in node: {anomaly_node}')

# Function to perform anomaly detection using an Approximate Nearest Neighbor (ANN) algorithm
def ann_algorithm(graph, embeddings, to_print=True, algo='ann', pred=[], node_to_index={}):    
    # Initialize an Annoy index for nearest neighbor search
    dimension = output_size  # Number of features in the vectors
    index = AnnoyIndex(dimension, 'euclidean')

    # Add vectors to the index
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)

    n_trees =  10  # Number of trees for the index
    search_k = -1  # -1 means use n_trees * n_trees
    num_neighbors = 4  # Number of neighbors to find 
    
    # Build the index
    index.build(n_trees)
    
    # Calculate distances to the nearest neighbors
    distances = [index.get_nns_by_item(i, num_neighbors, include_distances=True, search_k=search_k)[1] for i in range(embeddings.shape[0])]
    
    # Calculate anomaly scores
    anomaly_scores = np.mean(distances, axis=1)
    
    avg_distance = np.mean(anomaly_scores)
    std_distance = np.std(anomaly_scores)
    
    anomalies = [i for i in range(len(anomaly_scores)) 
                        if (anomaly_scores[i] > avg_distance + ann_threshold * std_distance) ]
    
    # Print the anomalies nodes
    list_nodes = list(graph.nodes)
    for anomaly in anomalies:
        anomaly_node_id = list_nodes[anomaly]
        if to_print:
            print_anomalies(graph, anomaly_node_id, "ann")
        graph.nodes[anomaly_node_id]["pred"] = True
        graph.nodes[anomaly_node_id]["ann_pred"] = True
        
        if algo == 'combined':
            graph.nodes[anomaly_node_id]["pred"] = graph.nodes[anomaly_node_id]["cluster_pred"] or graph.nodes[anomaly_node_id]["cluster"] == -1
            if graph.nodes[anomaly_node_id]["pred"]:
                print_anomalies(graph, anomaly_node_id, "ann")
       
        pred[node_to_index[anomaly_node_id]] = graph.nodes[anomaly_node_id]["pred"]
            
    # add the anomaly score to the history queue + check anomalies nodes
    for i, anomaly_score in enumerate(anomaly_scores):
        node_id = list_nodes[i]
        queue = graph.nodes[node_id]["anomaly_score_history"]
        
        if len(queue) == anomaly_score_history_size:
            avg_distance = np.mean(queue)
            std_distance = np.std(queue)
        
            if anomaly_score > avg_distance + ann_history_threshold * std_distance:
                if to_print:
                    print_anomalies(graph, node_id, "history")
                graph.nodes[node_id]["pred"] = True
                graph.nodes[node_id]["ann_pred"] = True
                
                if algo == 'combined':
                    graph.nodes[node_id]["pred"] = graph.nodes[node_id]["cluster_pred"] or graph.nodes[node_id]["cluster"] == -1
                    if graph.nodes[node_id]["pred"]:
                        print_anomalies(graph, node_id, "ann")
                    
                
                             
        if len(queue) >= anomaly_score_history_size:
            queue.pop(0)  # Remove the first element
        queue.append(anomaly_score)
    
                
    return anomalies
