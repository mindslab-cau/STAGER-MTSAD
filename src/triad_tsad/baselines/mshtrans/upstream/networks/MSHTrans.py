import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import copy

from torch_geometric.data import data as D
from torch.nn import Linear
from tqdm.auto import tqdm
from networks.MAHLayer import Multi_Adaptive_Hypergraph, SnapshotAdaptiveHypergraph
from networks.Layers import Bottleneck_Construct, SeriesDecomposition, SeasonTrendFusion, MSFusion, PositionalEncoding
from networks.HyperGraphConv import HypergraphConv

class MSHTrans(nn.Module):
    def __init__(self, args, device):
        super(MSHTrans, self).__init__()
        self.n_feats = args["n_feats"]
        self.window_size = args["window_size"]
        self.max_seq_length = self.window_size
        self.hyper_num = args["scale_num"] 
        self.lr = args["lr"]
        self.model_root = args["model_root"]
        self.device = device
        self.kernel_size  = args["pool_size_list"]
        self.conv_layers = Bottleneck_Construct(self.n_feats, args["pool_size_list"], self.n_feats)
        self.seq_length = [self.max_seq_length]
        for i in range(self.hyper_num - 1):
            self.seq_length.append(self.seq_length[-1] // args["pool_size_list"][i])
        self.hyper_head = args["head_num"]

        moving_average_windows = args.get("moving_average_windows")
        explicit_sizes = moving_average_windows is not None
        graph_class = SnapshotAdaptiveHypergraph if explicit_sizes else Multi_Adaptive_Hypergraph
        self.multi_adpive_hypergraph = graph_class(args, self.device)
        
        self.pos_encoder = PositionalEncoding(self.n_feats * 2, 0.1, self.window_size)
        
        self.hyconv_list = nn.ModuleList()
        for i in range (self.hyper_num):
            self.hyconv = nn.ModuleList()
            for j in range(self.hyper_head):
                self.hyconv.append(HypergraphConv(
                    self.n_feats * 2, self.n_feats * 2, explicit_sizes=explicit_sizes
                ))
            self.hyconv_list.append(self.hyconv)
            
        self.series_decomposition = nn.ModuleList()
        self.season_trend_fusion = nn.ModuleList()
        for i in range(self.hyper_num):
            seq_len = self.seq_length[i]
            self.series_decomposition.append(
                SeriesDecomposition(seq_len, self.n_feats * 2, moving_average_windows)
            )
            self.season_trend_fusion.append(SeasonTrendFusion(self.n_feats * 2, self.n_feats * 2))
            
        self.msfusion = MSFusion(self.n_feats * 2, args["pool_size_list"], self.n_feats * 2, self.seq_length)
        
        self.series_decomposition_d1 = SeriesDecomposition(
            self.max_seq_length, self.n_feats, moving_average_windows
        )
        self.series_decomposition_d2 = SeriesDecomposition(
            self.max_seq_length, args["n_feats"], moving_average_windows
        )
        self.hyconv_d1 = nn.ModuleList()
        for j in range(self.hyper_head):
            self.hyconv_d1.append(HypergraphConv(
                self.n_feats * 2, self.n_feats, explicit_sizes=explicit_sizes
            ))
        self.season_trend_fusion_d1 = SeasonTrendFusion(self.n_feats, self.n_feats)
        self.sigmoid = nn.Sigmoid()
        
        self.node_embedding_proj = Linear(args["d_model"], self.n_feats)
        
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.trained_epochs = 0
        
    
    
    def extract_downsample(self, x):
        window_x_list = [x]
        for i in range(self.hyper_num - 1):
            idx = [j * pow(self.kernel_size[i], i + 1) for j in range(self.seq_length[i + 1])]
            sequence = x[:, idx, :]
            window_x_list.append(sequence)

        return window_x_list
        
    def encoder(self, x, hyper_graph_indicies):
        window_ori_x = self.extract_downsample(x)
        
        seq_enc = self.conv_layers(x) 
        
        for i in range(self.hyper_num):
            seq_enc[i] = torch.concat([seq_enc[i], window_ori_x[i]], dim=-1)
            seq_enc[i] = self.pos_encoder(seq_enc[i])

        st_fusion_list = []
        for i in range(self.hyper_num):
            hyperedge_indices = torch.tensor(hyper_graph_indicies[i]).to(self.device) 
            
            node_value = seq_enc[i].permute(0,2,1).to(self.device)

            edge_indices, node_indices = hyperedge_indices[1], hyperedge_indices[0]

            num_edges = edge_indices.max().item() + 1
            num_nodes = node_value.size(2)  

            indices = torch.stack([edge_indices, node_indices])  
            values = torch.ones(edge_indices.size(0), device=node_value.device)
            adj_matrix = torch.sparse_coo_tensor(indices, values, (num_edges, num_nodes), device=node_value.device)

   
            node_value = node_value.permute(2, 0, 1).contiguous()

            edge_features = torch.sparse.mm(adj_matrix, node_value.view(num_nodes, -1)).view(num_edges, node_value.size(1), node_value.size(2))
            
            multi_head_hyconv = self.hyconv_list[i]
            output_list = []
            for j in range(self.hyper_head):
                output = multi_head_hyconv[j](seq_enc[i], hyperedge_indices, edge_features).permute(1, 0, 2)
                output_list.append(output)
            multi_head_node_emb = torch.mean(torch.stack(output_list, dim = -1), dim = -1)
            
            multi_head_node_emb = multi_head_node_emb + seq_enc[i]
            
            seasonality, trend = self.series_decomposition[i](multi_head_node_emb)
            
            st_fusion = self.season_trend_fusion[i](seq_enc[i], seasonality, trend) 
            st_fusion_list.append(st_fusion)
        fused_logits = self.msfusion(st_fusion_list)
        return fused_logits

    def decoder(self, x, fused_logits, hyperedge_index):
        edge_features = {}
        
        node_value = x.permute(0,2,1)
        hyperedge_index = hyperedge_index.to(self.device)

        edge_indices, node_indices = hyperedge_index[1], hyperedge_index[0]

        num_edges = edge_indices.max().item() + 1
        num_nodes = node_value.size(2) 

        indices = torch.stack([edge_indices, node_indices])  
        values = torch.ones(edge_indices.size(0), device=node_value.device)
        adj_matrix = torch.sparse_coo_tensor(indices, values, (num_edges, num_nodes), device=node_value.device)

        node_value = node_value.permute(2, 0, 1).contiguous()

        edge_features = torch.sparse.mm(adj_matrix, node_value.view(num_nodes, -1)).view(num_edges, node_value.size(1), node_value.size(2))
        
        z_sea_1, z_trend_1 = self.series_decomposition_d1(x) 
        
        input_hyconv = torch.concat([z_sea_1, fused_logits], dim=-1)
        input_hyconv = self.pos_encoder(input_hyconv)
        
        output_list = []
        for i in range(self.hyper_head):
            output = self.hyconv_d1[i](input_hyconv, hyperedge_index, edge_features).permute(1, 0, 2) 
            output_list.append(output)
        
        multi_head_node_emb = torch.mean(torch.stack(output_list, dim = -1), dim = -1)
        z_sea_2, z_trend_2 = self.series_decomposition_d2(multi_head_node_emb)
        
        z_trend_3 = z_trend_1 + z_trend_2
        results = self.season_trend_fusion_d1(x, z_sea_2, z_trend_3)
        
        results = self.sigmoid(results)
  
        return results
        
    
    def forward(self, x, hyper_graph_indicies, fused_hypergraph):
        fused_logits = self.encoder(x, hyper_graph_indicies)
        predict_logits = self.decoder(x, fused_logits, fused_hypergraph)
 
        return predict_logits
    
    def hyperedge_constraint(self, window_ori_x, hyper_graph_indicies, node_embedding_list, edge_embedding_list, edge_retain_list):

        loss_hyperedge_all_scale = 0.0
        loss_node_all_scale = 0.0
        for i in range(self.hyper_num):
            hyper_graph_index = torch.tensor(hyper_graph_indicies[i]).to(self.device)
            
            edge_embedding = edge_embedding_list[i][edge_retain_list[i], :]
            x = window_ori_x[i]
            node_value = x.permute(0,2,1)
            
            edge_indices, node_indices = hyper_graph_index[1], hyper_graph_index[0]

            num_edges = edge_indices.max().item() + 1
            num_nodes = node_value.size(2)  

            indices = torch.stack([edge_indices, node_indices])  
            values = torch.ones(edge_indices.size(0), device=node_value.device)
            adj_matrix = torch.sparse_coo_tensor(indices, values, (num_edges, num_nodes), device=node_value.device)

            node_value = node_value.permute(2, 0, 1).contiguous()
            edge_features = torch.sparse.mm(adj_matrix, node_value.view(num_nodes, -1)).view(num_edges, node_value.size(1), node_value.size(2))

            edge_features = torch.mean(edge_features, dim=1)

            loss_hyper = 0.0
            for k in range(edge_features.size(0)):
                for m in range(edge_features.size(0)):
                    inner_product = torch.sum(edge_features[k, :] * edge_features[m, :], dim=-1, keepdim=True)
                    norm_q_i = torch.norm(edge_features[k, :], dim=-1, keepdim=True)
                    norm_q_i = torch.clamp(norm_q_i, min=1e-4)
                    norm_q_j = torch.norm(edge_features[m, :], dim=-1, keepdim=True)
                    norm_q_j = torch.clamp(norm_q_j, min=1e-4)
                    alpha = inner_product / (norm_q_i * norm_q_j)


                    distan = torch.norm(edge_embedding[k, :] - edge_embedding[m, :], dim=0, keepdim=True) 

                    loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
                    loss_hyper = loss_hyper + torch.abs(torch.mean(loss_item))
                
            loss_hyper = loss_hyper / ((edge_features.size(0) + 1) ** 2)
            loss_hyperedge_all_scale = loss_hyperedge_all_scale + loss_hyper
        
            node_embedding = self.node_embedding_proj(node_embedding_list[i])
            x_i = torch.index_select(node_embedding, dim=0, index=hyper_graph_index[0])
            x_j = torch.index_select(edge_features, dim=0, index=hyper_graph_index[1])
            loss_node = abs(torch.mean(x_i - x_j))
            loss_node_all_scale = loss_node_all_scale + loss_node
            
        
        loss_hyperedge_all_scale = 0.1 * loss_hyperedge_all_scale
        
        return loss_hyperedge_all_scale, loss_node_all_scale
        
    def Laplacian_constraint(self, H, Z):
        A = H @ H.t()
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        
        L_expanded = L.unsqueeze(0).expand(Z.size(0), -1, -1)
        
        LZ = torch.bmm(L_expanded, Z)
        
        Z_T_LZ = torch.bmm(Z.transpose(1, 2), LZ)
        
        loss = torch.einsum('bii->b', Z_T_LZ)
        loss = torch.mean(loss)
        return loss
        
    def _compute_loss_terms(self, ori_window_data, hyper_graph_indicies, fused_hypergraph, edge_retain_list, mse_func):
        z = self(ori_window_data, hyper_graph_indicies, fused_hypergraph)
        tgt = ori_window_data

        rec_loss = torch.mean(mse_func(z, tgt))

        window_ori_x = self.extract_downsample(ori_window_data)
        node_embedding_list, edge_embedding_list = self.multi_adpive_hypergraph.get_embeddings()
        loss_hyperedge_all_scale, loss_node_all_scale = self.hyperedge_constraint(
            window_ori_x,
            hyper_graph_indicies,
            node_embedding_list,
            edge_embedding_list,
            edge_retain_list,
        )

        loss_laplacian = 0.0
        for i in range(self.hyper_num):
            H = torch.mm(node_embedding_list[i], edge_embedding_list[i].t())
            H = F.softmax(F.relu(self.multi_adpive_hypergraph.alpha * H))
            loss_laplacian = loss_laplacian + self.Laplacian_constraint(H, window_ori_x[i])

        loss_sum = rec_loss + loss_hyperedge_all_scale + loss_node_all_scale + loss_laplacian
        return {
            "total": loss_sum,
            "rec": rec_loss,
            "hyper": loss_hyperedge_all_scale,
            "node": loss_node_all_scale,
            "lap": loss_laplacian,
        }

    def _evaluate_loader(self, dataloader):
        if dataloader is None:
            return None

        mse_func = nn.MSELoss(reduction="none")
        nn.Module.train(self, False)
        batch_losses = []

        with torch.no_grad():
            hyper_graph_indicies, _, edge_retain_list, fused_hypergraph, _ = self.multi_adpive_hypergraph(train=False)
            batch_iterator = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Validation",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for d in batch_iterator:
                ori_window_data = d[0].to(self.device)
                loss_terms = self._compute_loss_terms(
                    ori_window_data,
                    hyper_graph_indicies,
                    fused_hypergraph,
                    edge_retain_list,
                    mse_func,
                )
                batch_losses.append(float(loss_terms["total"].detach().cpu().item()))
                batch_iterator.set_postfix(
                    val_total=f"{batch_losses[-1]:.4f}",
                    val_rec=f"{loss_terms['rec'].detach().cpu().item():.4f}",
                )

        return float(np.mean(batch_losses)) if batch_losses else None

    def train(self, args, dataloader, valid_loader=None):
        mse_func = nn.MSELoss(reduction="none")
        patience = max(0, int(args.get("patience", 10)))
        best_val_loss = None
        best_state_dict = None
        best_epoch = 0
        patience_count = 0

        epoch_iterator = tqdm(
            range(1, args["nb_epoch"] + 1),
            total=args["nb_epoch"],
            desc="MSHTrans training",
            unit="epoch",
            dynamic_ncols=True,
        )

        for epoch in epoch_iterator:
            logging.info("Training epoch: %s", epoch)
            nn.Module.train(self, True)

            epoch_loss_sum = 0.0
            batch_count = 0

            hyper_graph_indicies, _, edge_retain_list, fused_hypergraph, _ = self.multi_adpive_hypergraph()
            batch_iterator = tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"Epoch {epoch}/{args['nb_epoch']}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

            for d in batch_iterator:
                ori_window_data = d[0].to(self.device)
                loss_terms = self._compute_loss_terms(
                    ori_window_data,
                    hyper_graph_indicies,
                    fused_hypergraph,
                    edge_retain_list,
                    mse_func,
                )

                loss_sum = loss_terms["total"]
                epoch_loss_sum += float(loss_sum.detach().cpu().item())
                batch_count += 1

                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()

                batch_iterator.set_postfix(
                    total=f"{loss_sum.detach().cpu().item():.4f}",
                    rec=f"{loss_terms['rec'].detach().cpu().item():.4f}",
                    hyper=f"{loss_terms['hyper'].detach().cpu().item():.4f}",
                    node=f"{loss_terms['node'].detach().cpu().item():.4f}",
                    lap=f"{loss_terms['lap'].detach().cpu().item():.4f}",
                )

            self.scheduler.step()
            train_epoch_loss = epoch_loss_sum / max(batch_count, 1)
            valid_epoch_loss = self._evaluate_loader(valid_loader)
            should_stop = False

            if valid_epoch_loss is not None:
                logging.info(
                    "Epoch %s finished | train_loss=%.4f | valid_loss=%.4f",
                    epoch,
                    train_epoch_loss,
                    valid_epoch_loss,
                )
                if best_val_loss is None or valid_epoch_loss < best_val_loss:
                    best_val_loss = valid_epoch_loss
                    best_state_dict = copy.deepcopy(self.state_dict())
                    best_epoch = epoch
                    patience_count = 0
                else:
                    patience_count += 1
                    logging.info(
                        "Early stopping counter: %s/%s",
                        patience_count,
                        patience,
                    )
                    if patience > 0 and patience_count >= patience:
                        logging.info(
                            "Early stopping triggered at epoch %s. Restoring best epoch %s.",
                            epoch,
                            best_epoch,
                        )
                        should_stop = True
            else:
                logging.info("Epoch %s finished | train_loss=%.4f", epoch, train_epoch_loss)

            epoch_iterator.set_postfix(
                train=f"{train_epoch_loss:.4f}",
                valid=f"{valid_epoch_loss:.4f}" if valid_epoch_loss is not None else "n/a",
                patience=(
                    f"{patience_count}/{patience}"
                    if valid_epoch_loss is not None and patience > 0
                    else "n/a"
                ),
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )
            if should_stop:
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            self.trained_epochs = best_epoch
        else:
            self.trained_epochs = epoch

        
            
    
    def predict_prob(self, dataloader):
        mse_func = nn.MSELoss(reduction="none")
        
        
        with torch.no_grad():
            nn.Module.train(self, False)
            loss_steps = []
            loss_all_var_steps = []
            z_steps = []
            hyper_graph_indicies, H_list, edge_retain_list, fused_hypergraph, fused_retain_edge = self.multi_adpive_hypergraph(train=False) 
            
            batch_iterator = tqdm(
                dataloader,
                total=len(dataloader),
                desc="Predict",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for d in batch_iterator:
                ori_window_data =d[0].to(self.device)
                
                z = self(ori_window_data, hyper_graph_indicies, fused_hypergraph)
                tgt = ori_window_data
                loss = mse_func(z, tgt) 
                
                loss_all_var = loss
                z_all_var = z
                
                loss = torch.mean(loss, dim=-1) 

                loss = loss[:, -1]

                loss_steps.append(loss.detach().cpu().numpy())
                loss_all_var_steps.append(loss_all_var.detach().cpu().numpy())
                z_steps.append(z_all_var.detach().cpu().numpy())
            anomaly_score = np.concatenate(loss_steps)
            loss_all_var_steps = np.concatenate(loss_all_var_steps, axis=0)
            z_steps = np.concatenate(z_steps, axis=0)
        
        return anomaly_score, loss_all_var_steps, z_steps
                    
