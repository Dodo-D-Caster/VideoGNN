# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import EdgeConv, APPNP
from .GPSConv import GPSConv
from .dggModel import GCN_DGG
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .pretrain_K_predictor_val_classification import VAELikeKPredictor, load_pretrained_vaelike_kpredictor

class Attention(nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.linear = nn.Linear(in_features, 1) 

    def forward(self, x_agg, y_agg):
        combined_features = torch.cat((x_agg, y_agg), dim=-1)
        attention_scores = self.linear(combined_features).squeeze(-1) 
        attention_weights = F.softmax(attention_scores, dim=0) 

        return attention_weights

class AttentionFusion(nn.Module):
    def __init__(self, in_features):
        super(AttentionFusion, self).__init__()
        self.attention = Attention(in_features * 2)

    def forward(self, x_agg, y_agg):
        assert x_agg.shape == y_agg.shape, "x_agg and y_agg must have the same shape"
        attention_weights = self.attention(x_agg, y_agg)
        attention_weights = attention_weights.unsqueeze(-1)
        combined_x = attention_weights * x_agg + (1 - attention_weights) * y_agg
        return combined_x
    
class Update(nn.Module):
    def __init__(self, in_features, out_features):
        super(Update, self).__init__()
        self.linear = nn.Linear(in_features*2, out_features)
        self.activation = nn.ReLU()

    def forward(self, combined_representation, x):
        concatenated = torch.cat((x, combined_representation), dim=1)
        updated_x = self.linear(concatenated)
        updated_x = self.activation(updated_x)
        return updated_x

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x
    
class GraphGPS(torch.nn.Module):
    def __init__(self, num_node_features, out_node_features):
        super(GraphGPS, self).__init__()
        self.conv1 = GPSConv(channels=num_node_features, conv=GCNConv)
        self.fc = torch.nn.Linear(num_node_features, out_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.fc(x)
        return x


class EdgeAgg(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_node_features):
        super(EdgeAgg, self).__init__()
        fc1 = torch.nn.Linear(num_node_features*2, hidden_channels)
        fc2 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.conv1 = EdgeConv(fc1, 'add')
        self.conv2 = EdgeConv(fc2, 'add')
        self.fc = torch.nn.Linear(hidden_channels, out_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x


class VideoGNN(nn.Module):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, n=196):
        super(VideoGNN, self).__init__()
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.n = n
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.alpha = nn.Parameter(torch.ones(2), requires_grad=True)
        # self.GCNModel = GCN(num_node_features=in_channels, hidden_channels=in_channels//2, out_node_features=out_channels)
        self.EdgeModel = EdgeAgg(num_node_features=in_channels, hidden_channels=in_channels, out_node_features=out_channels)
        # self.APPNPModel = APPNP(K=10, alpha=0.1)
        # self.GPSModel = GraphGPS(num_node_features=in_channels, out_node_features=out_channels)
        self.inOutFC = nn.Linear(in_channels, out_channels)
        self.fusion = AttentionFusion(out_channels)
        self.update = Update(in_features=out_channels, out_features=out_channels)
        mapToKPredictor = nn.Linear(in_channels, 500)
        KPredictorModel = VAELikeKPredictor(500, 64, 32, self.k)
        """
        If you meet the CUDA OOM issue, you can load the pretrained KPredictor. This pretrained weight makes the predicted degree start from a small value.
        Or you can lower the batch size of GNN
        """
        # load_pretrained_vaelike_kpredictor(KPredictorModel, "./modules/gcn_lib/vaelike_kpredictor_flickr_best_val.pth")
        self.KPredictor = nn.Sequential(mapToKPredictor, KPredictorModel)

    def generate_mask(self, x_k_int, device):
        """
        this code is equal to the code:
        
        def generate_mask(self, x_k_int, device):
            with torch.no_grad():
                x_k_int.to(device)
                mask = torch.zeros(len(x_k_int)*self.k, dtype=torch.bool, device=device)
                indices = torch.cat([torch.arange(i * self.k, i * self.k + k, device=device) for i, k in enumerate(x_k_int)]).to(device)
                mask[indices] = True
            return mask
        
        but this version runs much faster in GPU as it eliminates the for-loop.
        """
        
        with torch.no_grad():
            x_k_int = x_k_int.to(device)
            N = len(x_k_int)
            K = self.k
            mask = torch.zeros(N * K, dtype=torch.bool, device=device)
            base_indices = torch.arange(N, device=device).repeat_interleave(K) * K 
            offset_indices = torch.arange(K, device=device).expand(N, -1).flatten() 
            valid_mask = offset_indices < x_k_int.repeat_interleave(K) 
            all_indices = (base_indices + offset_indices)[valid_mask] 
            mask[all_indices] = True 
        return mask
    
    def add_global_node(self, x, edge_index, device):
        N, C = x.shape
        global_node_feature = torch.zeros(1, C, device=device)
        x = torch.cat([x, global_node_feature], dim=0).to(device) 

        global_node_index = torch.tensor([[N] * N, list(range(N))], device=device)
        global_edges = torch.cat([global_node_index, global_node_index.flip(0)], dim=1).to(device)

        edge_index = torch.cat([edge_index, global_edges], dim=1).to(device)
        return x, edge_index
    
    def forward(self, x, relative_pos=None, T=228):
        B, C, H, W = x.shape
        N = H*W
                 
        x = x.reshape(B, C, -1, 1).contiguous()
        x_edge_index = self.dilated_knn_graph(x, None, relative_pos)
        
        y = x.squeeze(-1).permute(0,2,1).view(B // T, T, N, C) # x (B,C,N,1) -> (B,N,C) -> (NB, T, N, C)
        y = torch.roll(y, shifts=-1, dims=1)
        y = y.view(B, N, C).permute(0,2,1).unsqueeze(-1)
        y_edge_index = self.dilated_knn_graph(x, y, relative_pos)
        
        x = x.view(B*N, C)
        x_edge_index = x_edge_index.view(2,B*N*self.k)
        x_logits,_,_ = self.KPredictor(x)
        x_k_int = x_logits.argmax(dim=1)

        y = y.contiguous().view(B*N, C)
        y_edge_index = y_edge_index.view(2,B*N*self.k)
        
        x_data_list = []
        y_data_list = []
        x_k_int = x_k_int.to(x.device)
        masks = self.generate_mask(x_k_int, x.device)
        for bb in range(B):
            start_node = bb*N
            end_node = start_node + N
            start_edge = start_node*self.k
            end_edge = end_node*self.k
            mask = masks[start_edge:end_edge]
            
            x_batch = x[start_node:end_node]
            x_edge_index_batch = x_edge_index[:, start_edge:end_edge]
            # mask = self.generate_mask(x_k_int[start_node:end_node], x.device)
            x_edge_index_batch = x_edge_index_batch[:, mask]
            # x_batch, x_edge_index_batch = self.add_global_node(x_batch, x_edge_index_batch, device=x.device)
            x_data = Data(x=x_batch, edge_index=x_edge_index_batch)
            x_data_list.append(x_data)
            
            y_batch = y[start_node:end_node]
            y_edge_index_batch = y_edge_index[:, start_edge:end_edge]
            y_edge_index_batch = y_edge_index_batch[:,mask]
            # y_batch, y_edge_index_batch = self.add_global_node(y_batch, y_edge_index_batch, device=x.device)
            y_data = Data(x=y_batch, edge_index=y_edge_index_batch)
            y_data_list.append(y_data)
        
        """
        you can lower the batch_size here, to save the CUDA memory
        """
        x_loader = DataLoader(x_data_list, batch_size=T, shuffle=False)
        y_loader = DataLoader(y_data_list, batch_size=T, shuffle=False)
        # model = self.GCNModel
        model = self.EdgeModel
        # model = self.GPSModel
        # model = self.GCNIIModel_DGG
        # model = self.APPNPModel
            
        model.train()
        x_agg = []
        for data in x_loader:
            out = model(data)
            # out = model(data.x, data.edge_index)
            x_agg.append(out)
        x_agg = torch.cat(x_agg, dim=0)  # Shape: [B*N, C]
        
        y_agg = []
        for data in y_loader:
            out = model(data)
            # out = model(data.x, data.edge_index)
            y_agg.append(out)
        y_agg = torch.cat(y_agg, dim=0)  # Shape: [B*N, C]

        x_fused = self.fusion(x_agg, y_agg)
        x = self.update(x_fused, self.inOutFC(x))

        # add global node
        # x = x.view(B, N+1, 2*C, 1)[:, :N, :, :].permute(0, 2, 1, 3)
        # no global node 
        x = x.view(B, N, 2*C, 1).permute(0, 2, 1, 3)
        return x.reshape(B, -1, H, W).contiguous()



class ImageGNN(nn.Module):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, n=196):
        super(ImageGNN, self).__init__()
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.n = n
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.alpha = nn.Parameter(torch.ones(2), requires_grad=True)
        # self.GCNModel = GCN(num_node_features=in_channels, hidden_channels=in_channels//2, out_node_features=out_channels)
        self.EdgeModel = EdgeAgg(num_node_features=in_channels, hidden_channels=in_channels, out_node_features=out_channels)
        # self.APPNPModel = APPNP(K=10, alpha=0.1)
        # self.GPSModel = GraphGPS(num_node_features=in_channels, out_node_features=out_channels)
        # self.GCNModel_DGG = GCN_DGG(nfeat=in_channels, nhidden=in_channels, nlayers=2, nclass=out_channels)
        self.inOutFC = nn.Linear(in_channels, out_channels)
        self.fusion = AttentionFusion(out_channels)
        self.update = Update(in_features=out_channels, out_features=out_channels)
        mapToKPredictor = nn.Linear(in_channels, 500)
        KPredictorModel = VAELikeKPredictor(500, 64, 32, self.k)
        # load_pretrained_vaelike_kpredictor(KPredictorModel, "modules/gcn_lib/vaelike_kpredictor_flickr_best_val.pth")
        self.KPredictor = nn.Sequential(mapToKPredictor, KPredictorModel)

    def generate_mask(self, x_k_int, device):
        """
        this code is equal to the code:
        
        def generate_mask(self, x_k_int, device):
            with torch.no_grad():
                x_k_int.to(device)
                mask = torch.zeros(len(x_k_int)*self.k, dtype=torch.bool, device=device)
                indices = torch.cat([torch.arange(i * self.k, i * self.k + k, device=device) for i, k in enumerate(x_k_int)]).to(device)
                mask[indices] = True
            return mask
        
        but this version runs much faster in GPU as it eliminates the for-loop.
        """
        
        with torch.no_grad():
            x_k_int = x_k_int.to(device)
            N = len(x_k_int)
            K = self.k
            mask = torch.zeros(N * K, dtype=torch.bool, device=device)
            base_indices = torch.arange(N, device=device).repeat_interleave(K) * K 
            offset_indices = torch.arange(K, device=device).expand(N, -1).flatten() 
            valid_mask = offset_indices < x_k_int.repeat_interleave(K) 
            all_indices = (base_indices + offset_indices)[valid_mask] 
            mask[all_indices] = True 
        return mask
    
    def add_global_node(self, x, edge_index, device):
        N, C = x.shape
        global_node_feature = torch.zeros(1, C, device=device)
        x = torch.cat([x, global_node_feature], dim=0).to(device) 

        global_node_index = torch.tensor([[N] * N, list(range(N))], device=device)
        global_edges = torch.cat([global_node_index, global_node_index.flip(0)], dim=1).to(device)

        edge_index = torch.cat([edge_index, global_edges], dim=1).to(device)
        return x, edge_index
    
    def forward(self, x, relative_pos=None, T=-1):
        B, C, H, W = x.shape
        N = H*W
                 
        x = x.reshape(B, C, -1, 1).contiguous()
        x_edge_index = self.dilated_knn_graph(x, None, relative_pos)
        x = x.view(B*N, C)
        x_edge_index = x_edge_index.view(2,B*N*self.k)
        # GG
        x_logits,_,_ = self.KPredictor(x)
        x_k_int = x_logits.argmax(dim=1)

        # KNN
        # x_k_int = torch.full((B*N,), 5)
        
        x_data_list = []
        x_k_int = x_k_int.to(x.device)
        masks = self.generate_mask(x_k_int, x.device)
        for bb in range(B):
            start_node = bb*N
            end_node = start_node + N
            start_edge = start_node*self.k
            end_edge = end_node*self.k
            mask = masks[start_edge:end_edge]
            
            x_batch = x[start_node:end_node]
            x_edge_index_batch = x_edge_index[:, start_edge:end_edge]
            # mask = self.generate_mask(x_k_int[start_node:end_node], x.device)
            x_edge_index_batch = x_edge_index_batch[:, mask]
            # x_batch, x_edge_index_batch = self.add_global_node(x_batch, x_edge_index_batch, device=x.device)
            x_data = Data(x=x_batch, edge_index=x_edge_index_batch)
            x_data_list.append(x_data)
            
        x_loader = DataLoader(x_data_list, batch_size=T, shuffle=False)
        # model = self.GCNModel
        model = self.EdgeModel
        # model = self.GPSModel
        # model = self.GCNModel_DGG
        # model = self.APPNPModel
            
        model.train()
        x_agg = []
        for data in x_loader:
            out = model(data)
            # out = model(data.x, data.edge_index)
            
            # # DGG Setting
            # num_nodes = data.num_nodes
            # adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            # adj[data.edge_index[0], data.edge_index[1]] = 1.0
            
            # in_adj = adj.to_sparse().to(data.x.device)
            # out,_,_ = model(data.x, in_adj)
            
            x_agg.append(out)
        x_agg = torch.cat(x_agg, dim=0)  # Shape: [B*N, C]
        
        x = self.update(x_agg, self.inOutFC(x))

        # add global node
        # x = x.view(B, N+1, 2*C, 1)[:, :N, :, :].permute(0, 2, 1, 3)
        # no global node 
        x = x.view(B, N, 2*C, 1).permute(0, 2, 1, 3)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False, isVideo=True):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        if isVideo == True:
            self.graph_conv = VideoGNN(in_channels, in_channels * 2, kernel_size, dilation,
                                act, norm, bias, stochastic, epsilon, r, self.n)
        else:
            self.graph_conv = ImageGNN(in_channels, in_channels * 2, kernel_size, dilation,
                                act, norm, bias, stochastic, epsilon, r, self.n)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x, T=228):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos, T)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x