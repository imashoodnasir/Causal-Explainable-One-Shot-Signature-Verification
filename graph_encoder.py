import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GraphEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64):
        super(GraphEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)

def compute_embedding(graph_data):
    model = GraphEncoder()
    return model(graph_data.x, graph_data.edge_index)
