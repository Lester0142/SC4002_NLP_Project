import sys
import torch
import data_getter as data_getter
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F

word2vec_model = data_getter.load_restricted_w2v()

class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2, activationfn='tanh', og = False):
        super(SentimentRNN, self).__init__()
        self.embed_layer = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors), freeze = False, padding_idx=0)
        self.activation = nn.ReLU()
        # RNN layer
        self.rnn = nn.RNN(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout, 
            nonlinearity=activationfn, 
            bias=True
            )
    
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    # Inside your model's forward method:
    def forward(self, x, mask):
        
        x = self.embed_layer(x)
        # print(x)
        # Pack the padded sequence
        x = self.activation(x)
        mask = (x.sum(dim=2) != 0).float()
        # print('og mask')
        # print(mask)
        lengths = mask.sum(dim=1).int()  # Compute the lengths of the sequences (number of non-padded elements)
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        packed_out, hidden = self.rnn(packed_x)
        
        # Unpack the sequence
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        
        # Perform mean pooling only over the valid (non-padded) parts
        out = (out * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1, keepdim=True)  # Mean pooling
        # print(out)
        out = self.fc(out)  # Pass through the fully connected layer

        return out
    

class SentimentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2, bidirectional=True):
        super(SentimentGRU, self).__init__()
        self.embed_layer = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors), freeze=False, padding_idx=0)
        self.activation = nn.ReLU()
        
        # Batch normalization for embedding layer output
        self.embed_norm = nn.BatchNorm1d(input_size)

        # Initialize the GRU layer
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout is only applied if num_layers > 1
            bidirectional=bidirectional
        )

        # Batch normalization for GRU output
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.gru_norm = nn.BatchNorm1d(gru_output_size)
        
        # Fully connected layer
        self.fc = nn.Linear(gru_output_size, output_size)

    def forward(self, x, mask):
        x = self.embed_layer(x)
        
        # Apply batch normalization to embeddings (requires reshaping for BatchNorm1d)
        # x = x.transpose(1, 2)  # Transpose to (batch, features, sequence length)
        # x = self.embed_norm(x)
        # x = x.transpose(1, 2)  # Transpose back to (batch, sequence length, features)
        
        # temporary remove relu
        # x = self.activation(x)
        
        # Compute the mask for the padded sequences
        mask = (x.sum(dim=2) != 0).float()
        lengths = mask.sum(dim=1).int()
        
        # Pack the padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through the GRU
        packed_out, hidden = self.gru(packed_x)
        
        # Unpack the sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Apply mean pooling over valid parts only
        # out = (out * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
         
        # Apply max pooling instead of mean pooling
        out = out.masked_fill(mask.unsqueeze(2) == 0, -float('inf'))  # Mask padding tokens with -inf
        out, _ = out.max(dim=1)  # Take max along the sequence dimension

        # Apply batch normalization to GRU output
        # out = self.gru_norm(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2, activationfn='tanh', bidirectional=True):
        super(SentimentLSTM, self).__init__()
        
        # Embedding layer with batch normalization
        self.embed_layer = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors), freeze=False, padding_idx=0)
        self.embed_norm = nn.BatchNorm1d(input_size)
        self.activation = nn.ReLU()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            bias=True,
            batch_first=True, 
            dropout=dropout, 
            bidirectional=bidirectional
        )

        # Fully connected layer with batch normalization
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.lstm_norm = nn.BatchNorm1d(lstm_output_size)
        self.fc = nn.Linear(lstm_output_size, output_size)

    def forward(self, x, mask):
        # Embedding layer with batch normalization
        x = self.embed_layer(x)
        
        # Transpose for BatchNorm1d, which expects (batch, features, sequence_length)
        # x = x.transpose(1, 2)
        # x = self.embed_norm(x)
        # x = x.transpose(1, 2)
        
        x = self.activation(x)
        
        # Calculate mask for packed sequence
        mask = (x.sum(dim=2) != 0).float()
        lengths = mask.sum(dim=1).int()
        
        # Pack the padded sequence for the LSTM
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_out, hidden = self.lstm(packed_x)
        
        # Unpack the sequence
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        
        # Apply mean pooling over valid parts only
        out = (out * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
        # Apply batch normalization to the LSTM output
        # out = self.lstm_norm(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out
    

class SentimentCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, output_size, dropout=0.5):
        super(SentimentCNN, self).__init__()
        self.embed_layer = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors), freeze=False, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # Embed input tokens
        x = self.embed_layer(x).unsqueeze(1)  # Shape: (batch_size, 1, seq_len, embedding_dim)

        # Apply convolutional layers and activation
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x)).squeeze(3)  # Shape: (batch_size, num_filters, new_seq_len)
           
            # Adjust mask size to match conv_out sequence length
            if mask is not None:
                conv_mask = mask[:, :conv_out.size(2)]  
                conv_mask = conv_mask.unsqueeze(1).expand_as(conv_out) 
                conv_out = conv_out * conv_mask  

                # Mean pooling over valid (non-padded) parts
                valid_lengths = conv_mask.sum(dim=2) 
                pooled_out = conv_out.sum(dim=2) / (valid_lengths + 1e-8)  
            else:
                pooled_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  

            conv_outs.append(pooled_out)

        # Concatenate pooled outputs
        out = torch.cat(conv_outs, dim=1) 

        # Apply dropout and final linear layer
        out = self.dropout(out)
        out = self.fc(out)

        return out
    

def load_model(model_type, device):

    if model_type.strip().upper() == "RNN":
        model = SentimentRNN(word2vec_model.vector_size, 16, 1, 1, 0.6, og=True).to(device)
        model.load_state_dict(torch.load('./models_weight/sentiment_rnn_1hl_16_00005.pth'))
        batch_size = 48
    elif model_type.strip().upper() == "RNN_OOV":
        model = SentimentRNN(word2vec_model.vector_size, 16, 1, 1, 0.6).to(device)
        model.load_state_dict(torch.load('./models_weight/sentiment_rnn_1hl_16_00005_with_OOV.pth'))
        batch_size = 48
    elif model_type.strip().upper() == "GRU":
        model = SentimentGRU(word2vec_model.vector_size, 16, 1, 2, 0.8).to(device)
        model.load_state_dict(torch.load('./models_weight/sentiment_bigru_2hl_16_-05_with_OOV.pth'))
        batch_size = 128
    elif model_type.strip().upper() == "LSTM":
        model = SentimentLSTM(word2vec_model.vector_size, 64, 1, 3, 0.6).to(device)
        model.load_state_dict(torch.load('./models_weight/sentiment_bilstm_2hl_64_-05_with_OOV.pth'))
        batch_size = 32
    elif model_type.strip().upper() == "CNN":
        model = SentimentCNN(word2vec_model.vector_size, 100, [3, 4, 5], output_size=1, dropout=0.7).to(device)
        model.load_state_dict(torch.load('./models_weight/sentiment_cnn_activation_[3, 4, 5]_0001_with_OOV.pth', map_location=torch.device('cpu')))
        batch_size = 48

    return model, batch_size
