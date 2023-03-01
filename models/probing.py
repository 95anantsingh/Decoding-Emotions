import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbingModel(nn.Module):
    def __init__(self, fx_model_name, num_labels):
        super().__init__()
        
        self.fx_model = fx_model_name
        hidden_dim = 128

        #Input Dim = Batch * 12 * Seq_Len * 768
        if 'BASE' in fx_model_name:
            feature_dim = 768
        elif 'LARGE' in fx_model_name:
            feature_dim = 1024
        elif self.fx_model == 'GE2E':
            feature_dim = 256
            
        self.linear1 = nn.Linear(in_features = feature_dim, out_features = hidden_dim)
        self.linear2 = nn.Linear(in_features = hidden_dim, out_features = num_labels)

    def forward(self, x, lengths, layer):
        """
        padded_x: (B,T) padded LongTensor
        """
        # batch_size, n_layers, seq_len, n_features = x.size(0), x.size(1), x.size(2), x.size(3)
        
        if self.fx_model != 'GE2E':
            x = x[:, layer, :, :]
            x = torch.mean(x, dim = 1)

        x = F.relu(self.linear1(x))
        logits = self.linear(x)

        return logits