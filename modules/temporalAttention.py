import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter

import copy


class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 n_heads, 
                 attn_drop, 
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.residual = residual

        # Define weights
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        
        # Feedforward layer
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # Dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In: attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        N, T, C = inputs.shape  # Get batch size, number of time steps, and feature size

        # 1: Create position embeddings dynamically
        position_embeddings = torch.arange(T, device=inputs.device).unsqueeze(0).expand(N, -1).unsqueeze(-1)  # [N, T, 1]
        position_inputs = position_embeddings.float()  # Use float for addition

        # 2: Add position embeddings to input
        temporal_inputs = inputs + position_inputs  # [N, T, C]

        # 3: Query, Key based multi-head self-attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, C]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, C]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, C]

        # 4: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size, dim=2), dim=0)  # [hN, T, C/h]
        k_ = torch.cat(torch.split(k, split_size, dim=2), dim=0)  # [hN, T, C/h]
        v_ = torch.cat(torch.split(v, split_size, dim=2), dim=0)  # [hN, T, C/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (T ** 0.5)

        # 5: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 6: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, C/h]
        
        # print("1", outputs.shape)
        # outputs = torch.cat(torch.split(outputs, split_size, dim=0), dim=2)  # [N, T, C]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2)
        # print("2", outputs.shape)

        # 7: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)