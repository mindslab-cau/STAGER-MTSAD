import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_layers import GCNLayer


def parse_ms_windows(ms_windows):
    if ms_windows is None:
        return []
    if isinstance(ms_windows, int):
        return [ms_windows]
    if isinstance(ms_windows, str):
        text = ms_windows.strip()
        if text.lower() in {"", "none", "null", "raw", "single_scale", "single", "[]"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [int(window.strip()) for window in text.split(",") if window.strip()]
    return [int(window) for window in ms_windows]

class TemporalDAG(nn.Module):
    def __init__(self, num_features, feature_dim, adj_func, node_dim=None, use_bn=False, feature_encoder_type='multi_scale', tag_mixing=True, nonmix_method='per_linear', shift=2, tgnn_layers=2, tgnn_hidden_multiplier=2, temporal_pool='mean', ms_windows="5,10,20", graph_type='dag'):
        super(TemporalDAG, self).__init__()
        if graph_type not in {'dag', 'fc'}:
            raise ValueError(f"Invalid graph_type: {graph_type}. Choose from 'dag' or 'fc'.")
        feature_encoder_type = feature_encoder_type.lower()
        if feature_encoder_type in {'raw', 'single_scale'}:
            feature_encoder_type = 'multi_scale'
            ms_windows = []
        ms_windows = parse_ms_windows(ms_windows)

        self.num_features = num_features
        self.feature_dim = feature_dim
        self.tag_mixing = tag_mixing
        self.node_dim = node_dim if node_dim is not None else feature_dim
        self.temporal_pool = temporal_pool
        self.shift = shift
        self.graph_type = graph_type
        self.feature_encoder_type = feature_encoder_type
        self.ms_windows = ms_windows
        self.tgnn_layers = max(1, int(tgnn_layers))

        if feature_encoder_type == 'diagonal':
            self.feature_encoder = DiagonalEmbedding(feature_dim, self.node_dim)
        elif feature_encoder_type == 'multi_scale':
            # MultiScaleEmbedding (input_dim=4 for raw + 3 MAs)
            self.feature_encoder = MultiScaleEmbedding(
                hidden_dim=32, 
                output_dim=self.node_dim, 
                windows=self.ms_windows
            )
        else:
            raise ValueError(f"Unknown feature_encoder_type: {feature_encoder_type}")

        hidden_dim = int(self.node_dim * tgnn_hidden_multiplier)
        layers = []
        if self.tgnn_layers == 1:
            layers.append(
                GCNLayer(num_features, self.node_dim, self.node_dim, tag_mixing=self.tag_mixing, nonmix_method=nonmix_method)
            )
        else:
            layers.append(
                GCNLayer(num_features, self.node_dim, hidden_dim, tag_mixing=self.tag_mixing, nonmix_method=nonmix_method)
            )
            for _ in range(self.tgnn_layers - 2):
                layers.append(
                    GCNLayer(num_features, hidden_dim, hidden_dim, tag_mixing=self.tag_mixing, nonmix_method=nonmix_method)
                )
            layers.append(
                GCNLayer(num_features, hidden_dim, self.node_dim, tag_mixing=self.tag_mixing, nonmix_method=nonmix_method)
            )
        self.shared_gcn_layers = nn.ModuleList(layers)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(self.num_features, affine=False, track_running_stats=True) if self.use_bn else None

        # Adjacency matrix
        inverse = False
        if adj_func.endswith('_inverse'):
            if self.graph_type == 'fc':
                raise ValueError("Inverse adjacency is only meaningful for graph_type='dag'.")
            adj_func = adj_func.replace('_inverse', '')
            inverse = True

        if adj_func == 'sigmoid':
            self.register_buffer('adj', self.create_adjacency_matrix_sigmoid(feature_dim, shift=self.shift, inverse=inverse))
        elif adj_func == 'step':
            self.register_buffer('adj', self.create_adjacency_matrix_step(feature_dim, shift=self.shift, threshold=0, inverse=inverse))
        elif adj_func == 'relu':
            self.register_buffer('adj', self.create_adjacency_matrix_relu(feature_dim, shift=self.shift, inverse=inverse))
        elif adj_func == 'one':
            if inverse:
                raise ValueError("Inverse not supported for 'one' adjacency.")
            self.register_buffer('adj', self.create_adjacency_matrix_one(feature_dim))
        elif adj_func == 'seq':
            if inverse:
                raise ValueError("Inverse not supported for 'seq' adjacency.")
            self.register_buffer('adj', self.create_adjacency_matrix_seq(feature_dim))
        else:
            raise ValueError(f"Invalid adj_func value: {adj_func}. Choose from 'sigmoid', 'step', 'relu', 'one', or 'seq'.")

    def create_adjacency_matrix_sigmoid(self, window_size, shift=2, inverse=False):
        adj = torch.zeros(window_size, window_size)
        if self.graph_type == 'fc':
            indices = torch.arange(window_size).float()
            for i in range(window_size):
                row_values = shift - torch.abs(i - indices)
                adj[i, :] = torch.sigmoid(row_values)
        else:
            for i in range(window_size):
                if inverse:
                    length = window_size - i
                    row_values = torch.arange(shift - length + 1, shift + 1).float()
                    adj[i, i:] = torch.sigmoid(row_values.flip(0))
                else:
                    row_values = torch.arange(shift - i, shift + 1).float()
                    adj[i, :i + 1] = torch.sigmoid(row_values)
        deg = torch.diag(torch.pow(adj.sum(dim=1), -1)) 
        return deg @ adj

    def create_adjacency_matrix_step(self, window_size, shift=2, threshold=0, inverse=False):
        adj = torch.zeros(window_size, window_size)
        if self.graph_type == 'fc':
            indices = torch.arange(window_size).float()
            for i in range(window_size):
                row_values = shift - torch.abs(i - indices)
                adj[i, :] = (row_values > threshold).float()
        else:
            for i in range(window_size):
                if inverse:
                    length = window_size - i
                    row_values = torch.arange(shift - length + 1, shift + 1).float()
                    adj[i, i:] = (row_values.flip(0) > threshold).float()
                else:
                    row_values = torch.arange(shift - i, shift + 1).float()
                    adj[i, :i + 1] = (row_values > threshold).float()  # Step function
        deg = torch.diag(torch.pow(adj.sum(dim=1), -1))
        return deg @ adj

    def create_adjacency_matrix_relu(self, window_size, shift=2, inverse=False):
        adj = torch.zeros(window_size, window_size)
        if self.graph_type == 'fc':
            indices = torch.arange(window_size).float()
            for i in range(window_size):
                row_values = shift - torch.abs(i - indices)
                adj[i, :] = torch.relu(row_values)
        else:
            for i in range(window_size):
                if inverse:
                    length = window_size - i
                    row_values = torch.arange(shift - length + 1, shift + 1).float()
                    adj[i, i:] = torch.relu(row_values.flip(0))
                else:
                    row_values = torch.arange(shift - i, shift + 1).float()
                    adj[i, :i + 1] = torch.relu(row_values)  # ReLU-like transformation
        deg = torch.diag(torch.pow(adj.sum(dim=1), -1))
        return deg @ adj

    def create_adjacency_matrix_one(self, window_size):
        if self.graph_type == 'fc':
            adj = torch.ones(window_size, window_size)
        else:
            adj = torch.tril(torch.ones(window_size, window_size))
        deg = torch.diag(torch.pow(adj.sum(dim=1), -1))
        return deg @ adj

    def create_adjacency_matrix_seq(self, window_size, self_loop=True):
        adj = torch.zeros(window_size, window_size)
        if self_loop:
            adj.fill_diagonal_(1.0)
        
        indices = torch.arange(1, window_size)
        adj[indices, indices - 1] = 1.0
        if self.graph_type == 'fc':
            adj[indices - 1, indices] = 1.0
        
        deg = torch.diag(torch.pow(adj.sum(dim=1).clamp(min=1e-6), -1))
        return deg @ adj


    def forward(self, x):
        batch_size, num_features, feature_dim = x.size()

        # 1. Apply Feature Encoder (B, C, L, D)
        # x_enc shape: (B, C, L, D)
        x_enc = self.feature_encoder(x)


        # 2. Shape for GCN input
        # (B, C, L, D) -> (B, L, C, D)
        x_enc = x_enc.permute(0, 2, 1, 3)

        # (B, L, C*D) flatten
        x_concat = x_enc.reshape(batch_size, feature_dim, num_features * self.node_dim)

        # Apply TGNN GCN stack (skip activation for the last layer)
        out = x_concat
        for i, layer in enumerate(self.shared_gcn_layers):
            out = layer(out, self.adj)
            if i != len(self.shared_gcn_layers) - 1:
                out = F.relu(out)

        # Temporal pooling: (B, L, C*D) -> (B, 1, C*D)
        if self.temporal_pool == 'last':
            ema_out = out[:, -1:, :]  # (B, 1, C*D)
        else:  # 'mean'
            ema_out = out.mean(dim=1, keepdim=True)  # (B, 1, C*D)


        # Separate by channel and convert to final output shape
        # (B, 1, D*C) -> (B, C, D)
        ema_out = ema_out.view(batch_size, 1, num_features, self.node_dim)  # (B, 1, C, D)
        ema_out = ema_out.transpose(1, 2)  # (B, C, 1, D)
        ema_out = ema_out.squeeze(2)  # (B, C, D)

        if self.use_bn:
            ema_out = self.bn(ema_out) 
        
        return ema_out

class MultiScaleEmbedding(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=16, windows=None):
        """
        Args:
            hidden_dim: MLP intermediate dimension (recommended 32 for nonlinearity)
            output_dim: Final embedding dimension (16)
            windows: List of window sizes to calculate Moving Average
        """
        super().__init__()
        self.windows = [5, 10, 20] if windows is None else list(windows)
        self.input_dim = len(self.windows) + 1
        # MLP structure: 4 -> 32 -> ReLU -> 16
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def get_moving_average(self, x, window_size):
        """
        x: (Batch * F, 1, T)
        Calculate Moving Average, but use Padding to keep the input length.
        """
        pad_size = window_size // 2
        # T length to keep using padding (Same padding effect)
        # For even window sizes, processing can be complex, so odd window sizes are recommended or asymmetric padding
        # Here, we simply use replicate pad or constant pad
        x_padded = F.pad(x, (pad_size, window_size - 1 - pad_size), mode='replicate')
        
        # Calculate Moving Average using AvgPool1d (Stride=1 to keep length)
        ma = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
        return ma

    def forward(self, x):
        """
        Input x: (B, F, T) - (Batch, Multivariate, Time)
        Output: (B, F, T, output_dim)
        """
        B, F, T = x.shape
        
        # 1. Merge all features as independent samples
        # (B * F, 1, T) shape
        x_reshaped = x.view(B * F, 1, T)
        
        features = [x_reshaped] # Raw data
        
        # 2. Calculate Moving Average and add to list
        for w in self.windows:
            ma = self.get_moving_average(x_reshaped, w)
            features.append(ma)
            
        # 3. Concatenate along channel dimension (1) -> (B*F, 1+len(windows), T)
        # dim=1 is the channel (feature) dimension.
        x_aug = torch.cat(features, dim=1)
        
        # 4. MLP (B*F, T, Input_Dim)
        # Linear layer is the last dimension, so it should be in the shape of (Batch, ..., Input_Dim)
        x_aug = x_aug.permute(0, 2, 1) 
        
        # 5. MLP pass (Weight Sharing occurs)
        # All time steps (T) and all variables (F) use the same W
        out = self.mlp(x_aug) # Output shape: (B*F, T, Output_Dim)
        
        # 6. Restore original shape (B, F, ...)
        # Final Output: (B, F, T, Output_Dim)
        out = out.view(B, F, T, -1)
        
        return out


class DiagonalEmbedding(nn.Module):
    def __init__(self, feature_dim, node_dim):
        super(DiagonalEmbedding, self).__init__()
        self.feature_dim = feature_dim
        self.node_dim = node_dim
        # Initialize embedding matrix (feature_dim -> node_dim)
        self.embedding = nn.Parameter(torch.rand(feature_dim, node_dim))
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, x):
        # x: (Batch, Num_Features, Seq_Len)
        batch_size, num_features, seq_len = x.size()
        
        channel_embeddings = []
        for channel_idx in range(num_features):
            x_channel = x[:, channel_idx, :]  # (B, L)
            x_diag = torch.diag_embed(x_channel)  # (B, L, L)
            x_embedded = torch.matmul(x_diag, self.embedding)  # (B, L, D)
            channel_embeddings.append(x_embedded)
        
        # (B, C, L, D)
        return torch.stack(channel_embeddings, dim=1)
