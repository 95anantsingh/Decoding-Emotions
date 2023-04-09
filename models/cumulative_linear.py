import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeProbingLinear(nn.Module):
    def __init__(self, fx_model_name, num_labels, num_layers):
        super().__init__()

        self.fx_model = fx_model_name
        hidden_dim = 128

        #Input Dim = Batch * 12 * Seq_Len * 768
        if 'BASE' in fx_model_name:
            feature_dim = 768
        elif 'LARGE' in fx_model_name:
            feature_dim = 1024
            
        self.gamma = nn.Parameter(torch.ones(1))
        self.mixing_weights = nn.Parameter(torch.ones(num_layers))
        self.prob_weights = torch.ones(num_layers)
        
        self.linear1 = nn.Linear(in_features = feature_dim, out_features = hidden_dim)
        self.linear2 = nn.Linear(in_features = hidden_dim, out_features = num_labels)


    def forward(self, x, lengths, layer):

        x = x[:, :layer+1, :, :]

        batch_size, num_layers, seq_length, num_features = x.size(0), x.size(1), x.size(2), x.size(3)

        # Softmax 
        self.prob_weights = torch.softmax(self.mixing_weights, dim=-1)
        self.prob_weights = self.prob_weights.reshape(1, num_layers, 1, 1)

        # Weighted scaling
        x = torch.sum(x * self.prob_weights, dim=1)
        x = torch.mul(x,self.gamma)

        x = torch.mean(x, dim = 1)

        x = F.relu(self.linear1(x))
        logits = self.linear2(x)

        return logits
