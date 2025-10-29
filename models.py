import torch
import torch.nn as nn

class SNPEncoderCNN(nn.Module):
    """CNN encoder for SNP data"""
    def __init__(self, input_len, conv_channels=32, kernel_size=7, stride=2, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels*2, kernel_size=5, stride=stride, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_channels*2 * 16, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 2:
            # [B, L] -> [B, 1, L]
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            
            if x.size(1) != 1:
                raise ValueError(f"SNPEncoderCNN expects channel=1, but got {x.size(1)} in shape {tuple(x.shape)}")
        elif x.dim() == 4 and x.size(1) == 1 and x.size(2) == 1:
            
            x = x.squeeze(2)
        else:
            raise ValueError(f"SNPEncoderCNN expects [B,L]/[B,1,L]/[B,1,1,L], but got {tuple(x.shape)}")

        # Conv1d feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        return x


class MetaTraitFusion(nn.Module):
    """Attention-based meta fusion model
    """
    def __init__(self, num_traits=None, hidden_dim=64, **kwargs):
        super().__init__()
        
        if num_traits is None:
            if 'in_dim' in kwargs and kwargs['in_dim'] is not None:
                num_traits = int(kwargs['in_dim'])
            else:
                raise ValueError("MetaTraitFusion requires num_traits (or in_dim).")
        self.num_traits = int(num_traits)

        self.attn = nn.Sequential(
            nn.Linear(self.num_traits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_traits),
            nn.Softmax(dim=1)   
        )
        self.out = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, trait_preds: torch.Tensor):
        """
        trait_preds: [B, num_traits]
        back:
          pred:   [B] or [B]
          weights:[B, num_traits]
        """
        if trait_preds.dim() != 2 or trait_preds.size(1) != self.num_traits:
            raise ValueError(f"MetaTraitFusion expects input [B, {self.num_traits}], got {tuple(trait_preds.shape)}")

        weights = self.attn(trait_preds)                       # [B, num_traits]
        fused = torch.sum(weights * trait_preds, dim=1, keepdim=True)  # [B, 1]
        pred = self.out(fused).squeeze(1)                      # [B]
        return pred, weights
