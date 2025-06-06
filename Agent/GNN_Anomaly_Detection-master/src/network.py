import numpy as np
from annoy import AnnoyIndex

from config import network_threshold

BASE_LINE_AMOUNT = 100

class ANN():
    def __init__(self) -> None:
        self.index = None
    
    def create_index(self, dim):
        # Create an Annoy index
        dimension = dim  # Number of features in the vectors
        self.index = AnnoyIndex(dimension, 'euclidean')
        self.vectors = []

        # Build the index
        self.index.build(10)  # 10 trees
        self.num_neighbors = 4  # Number of neighbors to find
        self.number_of_vectors = 0

        self.mean_distances = np.array([])

    # Add a new vector to the annoy index
    def add_vector(self, vector):
        means_packet_length = [vector.fwd_packets_length/vector.fwd_packets_amount, 
                 vector.bwd_packets_length/vector.bwd_packets_amount]
        flags = list(vector.flags.values())
        attribute_values = flags + list(vector.__dict__.values())[8:] + means_packet_length
        vector_np = np.array(attribute_values)
        
        if self.index is None:
            self.create_index(len(vector_np))
        self.index.add_item(self.number_of_vectors, vector_np)
        self.vectors.append(vector_np)
        self.number_of_vectors += 1

        # Not calculating anomaly score for the first n vectors
        if self.number_of_vectors < BASE_LINE_AMOUNT:
            return 'base-line'

        # Calculate distances for the first n vectors
        if self.number_of_vectors == BASE_LINE_AMOUNT:
            # Calculate distances to the nearest neighbors
            distances = [self.index.get_nns_by_item(i, self.num_neighbors, include_distances=True, search_k=-1)[1][1:] 
                            for i in range(BASE_LINE_AMOUNT)]

            # Calculate anomaly scores
            self.mean_distances = np.mean(distances, axis=1)
                
            return 'base-line'
        
        # Calculate anomaly scores
        curr_vector_distances = self.index.get_nns_by_item(self.number_of_vectors-1, self.num_neighbors, include_distances=True, search_k=-1)[1]
        curr_vector_distances = curr_vector_distances[1:]
        curr_vector_mean_distance = np.mean(curr_vector_distances)
        all_vectors_means = np.mean(self.mean_distances)
        all_vectors_std = np.std(self.mean_distances)

        # Thresholding
        if curr_vector_mean_distance > all_vectors_means + network_threshold * all_vectors_std:
            return 'anomaly', curr_vector_mean_distance
        else:
            self.mean_distances = np.append(self.mean_distances, curr_vector_mean_distance)
            return 'normal', curr_vector_mean_distance
