import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout=0.0):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # Input: x (batch_size, seq_len, feature_dim)
        
        attention_output, _ = self.self_attention(x, x, x)  # Query, Key, Value
        # x = self.layer_norm(x + attention_output)
        x = F.relu(attention_output)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, num_heads=2):
        super(Encoder, self).__init__()
        self.linear_projection = nn.Linear(input_dim, feature_dim)
        self.transformer_layer = TransformerLayer(feature_dim, num_heads)

    def forward(self, x):
        # Input: x (num_samples, input_dim)
        projected_features = self.linear_projection(x)
        # projected_features: (num_samples, feature_dim)

        # transformer_input = projected_features.unsqueeze(1)  # Add sequence dimension
        # # transformer_input: (num_samples, 1, feature_dim)
        # 
        # encoded_features = self.transformer_layer(transformer_input)
        # # encoded_features: (num_samples, 1, feature_dim)
        # 
        # encoded_features = encoded_features.squeeze(1)  # Remove sequence dimension
        # # encoded_features: (num_samples, feature_dim)
        encoded_features = F.leaky_relu(projected_features)

        return encoded_features

class Decoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear_reconstruction = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        # Input: x (num_samples, feature_dim)
        reconstructed_output = self.linear_reconstruction(x)
        # reconstructed_output = F.leaky_relu(reconstructed_output);
        # reconstructed_output: (num_samples, output_dim)
        return reconstructed_output

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, feature_dim, output_dim):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, feature_dim)
        self.decoder = Decoder(feature_dim, output_dim)

    def forward(self, x):
        # Input: x (num_samples, input_dim)
        encoded_representation = self.encoder(x)
        # encoded_representation: (num_samples, feature_dim)
        reconstructed_output = self.decoder(encoded_representation)
        # reconstructed_output: (num_samples, output_dim)
        return reconstructed_output