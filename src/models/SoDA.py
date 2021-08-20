import torch
import torch.nn.functional as f
from .transformer_encoder import TransformerEncoder
from torch import nn


class SoDA(nn.Module):

    def __init__(self, dim_detection=4, dim_z=64, num_encoder_layers=6, num_attention_heads=8):
        super(SoDA, self).__init__()
        self.fc1 = nn.Linear(dim_detection, dim_z)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(dim_z, dim_z)
        self.layer_norm1 = nn.LayerNorm(dim_z)
        self.transformer_encoder = TransformerEncoder(num_layers=num_encoder_layers, dim_model=dim_z,
                                                      num_heads=num_attention_heads)
        self.fc3 = nn.Linear(dim_z, dim_z)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(dim_z, dim_z)
        self.tanH = nn.Tanh()

    def forward(self, x):
        z_0 = self.fc2(self.relu1(self.fc1(x)))
        z_n = self.transformer_encoder(z_0)
        z = self.tanH(self.fc4(self.relu2(self.fc3(z_n))))
        return z
