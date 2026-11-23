import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
except ImportError as exc:  # Keep importing the rest of the package possible.
    GATConv = GATv2Conv = GCNConv = None
    _GRAPH_IMPORT_ERROR = exc
else:
    _GRAPH_IMPORT_ERROR = None


class GCNLayer(nn.Module):
    def __init__(self, num_features, D_in, D_out, tag_mixing=True, nonmix_method="per_linear"):
        super(GCNLayer, self).__init__()
        if tag_mixing:
            self.linear = nn.Linear(num_features * D_in, num_features * D_out)
        elif nonmix_method == "per_linear":
            self.linear = PerChannelLinear(num_features, D_in, D_out)
        elif nonmix_method == "per_conv1d":
            self.linear = PerChannelConv1d(num_features, D_in, D_out)
        else:
            raise ValueError(
                f"Unknown nonmix_method: {nonmix_method}. "
                "Choose from 'per_linear' or 'per_conv1d'."
            )

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.linear(out)
        return out


class PerChannelLinear(nn.Module):
    def __init__(self, C: int, D_in: int, D_out: int, bias: bool = True):
        super().__init__()
        self.C = C
        self.linears = nn.ModuleList([nn.Linear(D_in, D_out, bias=bias) for _ in range(C)])

    def forward(self, x):
        B, L, _ = x.shape
        C = self.C
        D_in = self.linears[0].in_features
        x = x.view(B, L, C, D_in)
        ys = [self.linears[c](x[:, :, c, :]) for c in range(C)]
        y = torch.stack(ys, dim=2)
        return y.reshape(B, L, -1)


class PerChannelConv1d(nn.Module):
    def __init__(self, C: int, D_in: int, D_out: int, bias: bool = True):
        super().__init__()
        self.C = C
        self.D_in = D_in
        self.D_out = D_out

        self.proj = nn.Conv1d(
            in_channels=C * D_in,
            out_channels=C * D_out,
            kernel_size=1,
            groups=C,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        y = self.proj(x)
        y = y.transpose(1, 2).contiguous()
        return y


class FeatureGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        gnn_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.0,
        feature_residual: bool = False,
        edge_layout: str = "legacy",
    ):
        super().__init__()
        if _GRAPH_IMPORT_ERROR is not None:
            raise ImportError(
                "TriAD feature-graph layers require the optional "
                "'torch-geometric' dependency."
            ) from _GRAPH_IMPORT_ERROR
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if gnn_type not in {"gcn", "gat", "gatv2"}:
            raise ValueError("gnn_type must be one of 'gcn', 'gat', 'gatv2'.")

        self.num_nodes = num_nodes
        self.dropout_p = dropout
        self.gnn_type = gnn_type
        self.heads = heads
        self.feature_residual = feature_residual
        self.edge_layout = edge_layout

        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        self.register_buffer("base_edge_index", edge_index, persistent=False)

        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(out_dim)

        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activ = nn.GELU()

        for layer_idx in range(num_layers):
            in_channels = dims[layer_idx]
            out_channels = dims[layer_idx + 1]

            if gnn_type == "gcn":
                conv = GCNConv(in_channels, out_channels, add_self_loops=True, normalize=True)
            elif gnn_type == "gat":
                conv = GATConv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True,
                )
            elif gnn_type == "gatv2":
                conv = GATv2Conv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True,
                )
            else:
                raise ValueError(f"Invalid gnn_type: {gnn_type}. Choose from 'gcn', 'gat', 'gatv2'.")

            self.convs.append(conv)

    def expand_edge_index(self, batch_size: int, device: torch.device) -> torch.Tensor:
        base = self.base_edge_index.to(device)
        if batch_size == 1:
            return base

        offset = torch.arange(batch_size, device=device).view(-1, 1, 1) * self.num_nodes
        expanded = base.view(1, 2, -1) + offset
        if self.edge_layout == "per_sample":
            # Keep the source and target rows separate across samples.
            return expanded.permute(1, 0, 2).reshape(2, -1)
        return expanded.view(2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feat_dim = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Input node number ({num_nodes}) != defined value ({self.num_nodes})")

        loop_edge_index = self.expand_edge_index(batch_size, x.device)
        h = x.view(batch_size * num_nodes, feat_dim)

        for idx, conv in enumerate(self.convs):
            h = conv(h, loop_edge_index)
            if idx < len(self.convs) - 1:
                h = self.activ(h)
                if self.dropout is not None:
                    h = self.dropout(h)

        h = h.view(batch_size, num_nodes, -1)

        if self.feature_residual and h.shape == x.shape:
            h = h + x

        return h
