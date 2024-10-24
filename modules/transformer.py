import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, input_size, num_layers=1, num_heads=8, hidden_size=512, dropout=0.3, dim_feedforward=2048, 
                 debug=False, num_classes=-1):
        super(TransformerLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.debug = debug
        
        # Positional encoding for the Transformer
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Optional output layer if num_classes is defined
        if num_classes > 0:
            self.fc_out = nn.Linear(input_size, num_classes)
        else:
            self.fc_out = None

    def forward(self, src_feats, src_lens=None, hidden=None):
        """
        Args:
            - src_feats: (max_src_len, batch_size, D)
            - src_lens: (batch_size) -> optional, but we won't need it as much for Transformer
            
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size)
        """
        # Apply positional encoding
        src_feats = self.pos_encoder(src_feats)
        
        # Pass through the transformer encoder
        transformer_outputs = self.transformer_encoder(src_feats)
        
        # Optional final prediction
        if self.fc_out:
            predictions = self.fc_out(transformer_outputs)
        else:
            predictions = transformer_outputs
        
        return {
            "predictions": predictions,
            "hidden": None  # No need for hidden state in Transformer
        }

        
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings for relative positions
        self.rel_pos_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # Get the sequence length and batch size
        batch_size, seq_len, _ = x.size()
        
        # Create a tensor that holds the relative positions
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Pass through the embedding layer for relative positional encodings
        pos_emb = self.rel_pos_embeddings(positions)
        
        # Add relative positional encoding to input and apply dropout
        x = x + pos_emb
        return self.dropout(x)
