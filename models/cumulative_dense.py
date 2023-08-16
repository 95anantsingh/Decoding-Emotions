import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeProbingDense(nn.Module):
    def __init__(self, fx_model_name, num_labels, num_layers, device):
        super().__init__()
        
        self.fx_model = fx_model_name
        self.device = device
        hidden_dim = 256
        
        if 'BASE' in fx_model_name:
            feature_dim = 768
        elif 'LARGE' in fx_model_name:
            feature_dim = 1024           

        self.gamma = nn.Parameter(torch.ones(1))
        self.mixing_weights = nn.Parameter(torch.ones(num_layers))
        self.prob_weights = torch.ones(num_layers)
        
        # Input Dim = Batch_Size * Seq_Len * Feature_Dim
        self.cnn_layer1 = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=1)
        self.cnn_layer2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.2) # not used yet

        self.linear = nn.Linear(in_features = hidden_dim, out_features = num_labels)


    def forward(self, x, lengths, layer):

        x = x = x[:, :layer+1, :, :]

        batch_size, num_layers, seq_length, num_features = x.size(0), x.size(1), x.size(2), x.size(3)

        # Softmax 
        self.prob_weights = torch.softmax(self.mixing_weights, dim=-1)
        self.prob_weights = self.prob_weights.reshape(1, num_layers, 1, 1)

        # Weighted scaling
        x = torch.sum(x * self.prob_weights, dim=1)
        x = torch.mul(x,self.gamma)

        # Pass through CNN
        x = x.transpose(1,2) #now dimension is batch_size * num_features * seq_length
        x = F.relu(self.cnn_layer1(x))
        x = F.relu(self.cnn_layer2(x))
        x = x.transpose(1,2) #now dimension is batch_size * seq_length * num_features

        #Do global average over time sequence
        global_avg = torch.tensor([]).to(self.device)
        for i in range(batch_size):
            mean_vector = torch.mean(x[i,:lengths[i],:], dim = 0)
            mean_vector = mean_vector.reshape(1,-1)
            global_avg = torch.cat((global_avg, mean_vector))

        logits = self.linear(global_avg)

        return logits