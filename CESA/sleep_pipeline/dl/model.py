"""Deep-learning models for sleep staging.

Architectures
-------------
1. **SleepStagingModel** -- single-epoch CNN classifier.
2. **SleepStagingLSTM** -- CNN + bidirectional LSTM for sequence scoring.
3. **SleepAttentionModel** -- CNN + multi-head self-attention (lighter).
4. **SleepTransformer** -- CNN + Transformer encoder (full sequential).

Models 3 and 4 expose attention weights for interpretability.

All models are *skeletons* -- they must be trained on a labelled cohort.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SleepCNNBlock(nn.Module):
    """1-D temporal CNN feature extractor."""

    def __init__(self, in_channels: int, base_filters: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=50, stride=6, padding=22),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),

            nn.Conv1d(base_filters, base_filters * 2, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(base_filters * 2, base_filters * 2, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_features = base_filters * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C, T) -> (batch, out_features)"""
        return self.conv(x).squeeze(-1)


class SleepStagingModel(nn.Module):
    """Single-epoch CNN classifier.

    Parameters
    ----------
    n_channels : int
        Number of input signal channels (1 for EEG only, 2-3 with EOG/EMG).
    n_classes : int
        Number of output classes (5 for standard sleep staging).
    base_filters : int
        Base number of CNN filters.
    dropout : float
        Dropout rate before the final linear layer.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 5,
        base_filters: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.cnn = SleepCNNBlock(n_channels, base_filters)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.cnn.out_features, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C, T) -> (batch, n_classes) logits."""
        features = self.cnn(x)
        return self.classifier(features)


class SleepStagingLSTM(nn.Module):
    """CNN + bidirectional LSTM for sequence-level scoring.

    Processes a *sequence* of consecutive epochs to capture temporal
    context (e.g. NREM-REM cycles).

    Parameters
    ----------
    n_channels : int
        Signal channels per epoch.
    n_classes : int
        Output classes.
    base_filters : int
        CNN filter base.
    lstm_hidden : int
        LSTM hidden dimension.
    lstm_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 5,
        base_filters: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.cnn = SleepCNNBlock(n_channels, base_filters)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, C, T) -> (batch, seq_len, n_classes) logits.

        For single-epoch inference, use seq_len=1.
        """
        batch, seq_len, C, T = x.shape
        x_flat = x.view(batch * seq_len, C, T)
        cnn_out = self.cnn(x_flat)  # (batch*seq_len, feat)
        cnn_out = cnn_out.view(batch, seq_len, -1)  # (batch, seq_len, feat)

        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, 2*hidden)
        logits = self.classifier(lstm_out)  # (batch, seq_len, n_classes)
        return logits


# =========================================================================
# Positional encoding (shared by Attention and Transformer models)
# =========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence models."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# =========================================================================
# SleepAttentionModel -- CNN + single multi-head attention
# =========================================================================

class SleepAttentionModel(nn.Module):
    """CNN feature extractor + multi-head self-attention for sequence scoring.

    Lighter than a full Transformer: only one attention layer, but still
    captures which neighbouring epochs influence the current prediction.

    Attention weights are extractable via ``forward(..., return_attention=True)``.

    Parameters
    ----------
    n_channels : int
        Signal channels per epoch.
    n_classes : int
        Output classes.
    base_filters : int
        CNN base filters.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 5,
        base_filters: int = 64,
        n_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = SleepCNNBlock(n_channels, base_filters)
        d_model = self.cnn.out_features
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, C, T) -> (batch, seq_len, n_classes) logits.

        If *return_attention* is True, also returns (batch, n_heads, seq_len, seq_len).
        """
        batch, seq_len, C, T = x.shape
        x_flat = x.view(batch * seq_len, C, T)
        cnn_out = self.cnn(x_flat).view(batch, seq_len, -1)

        cnn_out = self.pos_enc(cnn_out)
        attn_out, attn_weights = self.attention(cnn_out, cnn_out, cnn_out)
        out = self.norm(cnn_out + attn_out)

        logits = self.classifier(out)

        if return_attention:
            return logits, attn_weights
        return logits


# =========================================================================
# SleepTransformer -- CNN + full Transformer encoder
# =========================================================================

class SleepTransformer(nn.Module):
    """CNN feature extractor + Transformer encoder for sequence-level scoring.

    Architecture:
    1. CNN extracts per-epoch feature vectors.
    2. Sinusoidal positional encoding adds temporal information.
    3. Transformer encoder (N layers, multi-head attention) captures
       inter-epoch dependencies via self-attention.
    4. Linear classification head per epoch.

    Attention weights from the last layer are extractable for
    interpretability.

    Parameters
    ----------
    n_channels : int
        Signal channels per epoch.
    n_classes : int
        Output classes.
    base_filters : int
        CNN base filters.
    n_heads : int
        Number of attention heads per Transformer layer.
    n_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        Feedforward dimension in each Transformer layer.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 5,
        base_filters: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = SleepCNNBlock(n_channels, base_filters)
        d_model = self.cnn.out_features
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (batch, seq_len, C, T) -> (batch, seq_len, n_classes) logits."""
        batch, seq_len, C, T = x.shape
        x_flat = x.view(batch * seq_len, C, T)
        cnn_out = self.cnn(x_flat).view(batch, seq_len, -1)

        cnn_out = self.pos_enc(cnn_out)
        encoded = self.transformer_encoder(cnn_out)
        logits = self.classifier(encoded)

        if return_attention:
            return logits, self._extract_attention(cnn_out)
        return logits

    def _extract_attention(self, src: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract attention weights from the last encoder layer."""
        try:
            last_layer = self.transformer_encoder.layers[-1]
            with torch.no_grad():
                _, attn = last_layer.self_attn(src, src, src, need_weights=True)
            return attn
        except Exception:
            return None
