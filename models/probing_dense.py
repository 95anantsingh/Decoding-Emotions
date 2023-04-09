
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbingDense(nn.Module):
    def __init__(self, fx_model_name, num_labels, device):
        super().__init__()
        
        self.fx_model = fx_model_name
        self.device = device
        hidden_dim = 256
        
        if 'BASE' in fx_model_name:
            feature_dim = 768
        elif 'LARGE' in fx_model_name:
            feature_dim = 1024
           
        # Input Dim = Batch_Size * Seq_Len * Feature_Dim
        self.cnn_layer1 = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=1)
        self.cnn_layer2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.2) # not used yet

        self.linear = nn.Linear(in_features = hidden_dim, out_features = num_labels)


    def forward(self, x, lengths, layer):
        """
        padded_x: (B,T) padded LongTensor
        """
        
        batch_size, num_layers, seq_length, num_features = x.size(0), x.size(1), x.size(2), x.size(3)
        
        x = x[:, layer, :, :]

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

