import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class AdaptiveGraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, num_nodes, embed_dim, num_time_labels):
        super(AdaptiveGraphConvolution, self).__init__(aggr='add') # "Add" aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_time_labels = num_time_labels
        
        # Node Embedding Matrix (E)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))

        # Time-Varying Weight Matrix (W_time)
        # One W per time label to map embeddings
        self.time_weights = nn.Parameter(torch.randn(num_time_labels, embed_dim, embed_dim))

        # Weight Generation Parameter (Theta)
        # Maps from embedding space to weight space
        self.weight_generator = nn.Linear(embed_dim, in_channels * out_channels)
        self.bias_generator = nn.Linear(embed_dim, out_channels)

    def forward(self, x, time_idx):
        """
        x: Node features of shape [batch_size, num_nodes, in_channels]
        time_idx: Time labels of shape [batch_size], e.g., 0 for morning, 1 for evening...
        """
        batch_size = x.size(0)
        
        # 1. Generate Dynamic Adaptive Adjacency Matrix
        # A_tilde = Softmax(ReLU(E * W_time * E^T))
        
        # Get the specific W_time for each sample in the batch: [batch_size, embed_dim, embed_dim]
        W_t = self.time_weights[time_idx]
        
        # E shape: [num_nodes, embed_dim] -> Expand to [batch_size, num_nodes, embed_dim]
        E_batch = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # E_Wt = E * W_time: [batch_size, num_nodes, embed_dim]
        E_Wt = torch.bmm(E_batch, W_t)
        
        # A_raw = E_Wt * E^T: [batch_size, num_nodes, num_nodes]
        A_raw = torch.bmm(E_Wt, E_batch.transpose(1, 2))
        
        # Softmax over neighbors
        A_tilde = F.softmax(F.relu(A_raw), dim=-1)
        
        # 2. Generate Node-Specific Weights (Static per node)
        # W_v = Theta * e_v. W has shape [num_nodes, in_channels * out_channels]
        weights = self.weight_generator(self.node_embeddings)
        weights = weights.view(self.num_nodes, self.in_channels, self.out_channels)
        bias = self.bias_generator(self.node_embeddings) # [num_nodes, out_channels]
        
        # 3. Apply Node-Specific Weights
        # Transform x: [batch_size, num_nodes, in_channels] -> [batch_size, num_nodes, out_channels]
        out = torch.einsum('bni,nio->bno', x, weights)
        
        # 4. Graph Convolution with Dynamic Adaptive Matrix
        # H^(l+1) = A_tilde * H^(l) * W_v
        out = torch.bmm(A_tilde, out)
        
        out = out + bias
        return F.relu(out)

class AH_GNN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, embed_dim, num_time_labels, num_layers=2):
        super(AH_GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(AdaptiveGraphConvolution(in_channels, hidden_channels, num_nodes, embed_dim, num_time_labels))
        for _ in range(num_layers - 1):
            self.layers.append(AdaptiveGraphConvolution(hidden_channels, hidden_channels, num_nodes, embed_dim, num_time_labels))
            
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, Z=None, time_idx=None, A_static=None):
        # x: [batch_size, num_nodes, in_channels]
        # Robust handling of arguments depending on how it's called
        if time_idx is None:
            # Called as model(x, time_idx)
            time_idx = Z
        for layer in self.layers:
            x = layer(x, time_idx)
        
        # Output prediction
        out = self.fc(x)
        return out

if __name__ == "__main__":
    # Test the model
    batch_size = 32
    num_nodes = 100
    in_channels = 12
    hidden_channels = 64
    out_channels = 1
    embed_dim = 10
    num_time_labels = 4 # e.g., night, rush_morning, rush_evening, normal
    
    model = AH_GNN(num_nodes, in_channels, hidden_channels, out_channels, embed_dim, num_time_labels)
    
    # Dummy input
    x = torch.randn(batch_size, num_nodes, in_channels)
    time_idx = torch.randint(0, num_time_labels, (batch_size,))
    
    output = model(x, time_idx)
    print("Dynamic AH-GNN Output Shape:", output.shape)
    assert output.shape == (batch_size, num_nodes, out_channels), "Shape mismatch"
    print("Dynamic AH-GNN forward pass successful!")

