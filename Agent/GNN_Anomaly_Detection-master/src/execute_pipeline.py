from visualization import plot_embeddings
from combined_algo import check_anomalies
from ann import ann_algorithm
from tri_graph import TriGraph
from clustering import check_all_anomalies, clustering_algorithm

def execute_pipeline(tri_graph:TriGraph, algo:str, plot:bool, pred=[], node_to_index={}):
    # print("Checking anomalies...")
    embeddings = tri_graph.create_embeddings()
    if algo == 'clustering' or algo == 'combined':
        cluster_embeddings = embeddings.detach().numpy()
        clusters = clustering_algorithm(cluster_embeddings)
        # check_all_anomalies(tri_graph.graph, cluster_embeddings, clusters, pred, node_to_index, algo != 'combined')
        check_all_anomalies(tri_graph.graph, cluster_embeddings, clusters, pred, node_to_index, True)
    if algo == 'ann' or algo == 'combined':
        # ann_algorithm(tri_graph.graph, embeddings.detach().numpy(), algo != 'combined', algo)
        ann_algorithm(tri_graph.graph, embeddings.detach().numpy(), True, algo, pred, node_to_index)
    # if algo == 'combined':
    #     check_anomalies(tri_graph.graph)

    if plot:
        tri_graph.visualize_directed_graph()
        plot_embeddings(embeddings, tri_graph.graph)