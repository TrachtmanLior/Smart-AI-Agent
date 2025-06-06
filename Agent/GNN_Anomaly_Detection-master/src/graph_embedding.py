import numpy as np
import torch
from torch_geometric.nn import GCNConv

from config import features, hidden_size, output_size

# Define a Graph Convolutional Network (GCN) model for generating embeddings
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

def create_embeddings(self):
        # Convert node features to PyTorch tensors
        node_features = torch.FloatTensor([list([self.graph.nodes[node][feature]/self.graph.nodes[node]['flows'] for feature in features]) 
                                           for node in self.graph.nodes])

        # Convert edges to PyTorch tensors
        node_to_index = {node: i for i, node in enumerate(self.graph.nodes())}
        edges = [(node_to_index[u], node_to_index[v]) for u, v in self.graph.edges()]

        # Convert the list of edges to a NumPy array
        edges_array = np.array(edges)

        # Convert the NumPy array to a PyTorch tensor
        edge_index = torch.tensor(edges_array, dtype=torch.long).t().contiguous()

        # Initialize the neural network
        num_features = len(node_features[0])
        
        if self.gcn_model is None:
            self.gcn_model = GCN(num_features, hidden_size, output_size)
        self.gcn_model.train()
        self.gcn_model.eval()

        embeddings = self.gcn_model(node_features, edge_index)
                
        return embeddings
