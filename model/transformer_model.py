import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """
    1-D convolutional stem: projects raw 468-dim CSI features into d_model
    while capturing local temporal patterns before self-attention.
    Inspired by modern vision Transformers applied to time-series.
    """
    def __init__(self, input_dim: int, d_model: int, kernel_size: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  (B, C, T) for Conv1d  →  (B, T, d_model)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings (better than sinusoidal for short CSI windows)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return self.dropout(x + self.pe(pos))


class CSITransformer(nn.Module):
    """
    Transformer encoder for Wi-Fi CSI Human Activity Recognition.

    Architecture mirrors the best-performing config from the repository
    experiments, but replaces LSTM with multi-head self-attention to
    capture long-range temporal dependencies across the 468-feature sequence.

    Input  : (batch, seq_len, 468)   – preprocessed CSI window
    Output : (batch, num_classes)    – raw logits
    """

    def __init__(
        self,
        input_dim:     int   = 468,
        d_model:       int   = 64,        # CPU-friendly; upgrade to 128+ with GPU
        nhead:         int   = 4,
        num_layers:    int   = 2,
        num_classes:   int   = 7,
        dim_feedforward: int = 128,
        dropout:       float = 0.2,
        max_seq_len:   int   = 512,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model

        # Convolutional feature extractor
        self.stem = ConvStem(input_dim, d_model)

        # Learnable positional encoding
        self.pos_enc = LearnablePositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN for more stable training
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # mean + max pooling concat
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 468)
        x = self.stem(x)          # (B, T, d_model)
        x = self.pos_enc(x)       # (B, T, d_model)
        x = self.encoder(x)       # (B, T, d_model)

        # Aggregation: concat global mean + global max → richer representation
        mean_pool = x.mean(dim=1)                 # (B, d_model)
        max_pool  = x.max(dim=1).values           # (B, d_model)
        pooled    = torch.cat([mean_pool, max_pool], dim=1)  # (B, d_model*2)

        return self.classifier(pooled)            # (B, num_classes)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities – convenience for inference."""
        return F.softmax(self(x), dim=1)
