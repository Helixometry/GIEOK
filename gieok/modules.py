"""Core modules for GIEOK."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAdapter(nn.Module):
    """
    Maps speech features to K speech tokens using Conv1d + adaptive pooling.

    Args:
        input_dim (int): Input dimension (speech encoder output)
        output_dim (int): Output token dimension
        n_tokens (int): Number of output tokens (K)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, n_tokens: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.AdaptiveMaxPool1d(n_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert speech features to speech tokens."""
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.layers(x)  # (batch, output_dim, n_tokens)
        return x.transpose(1, 2)  # (batch, n_tokens, output_dim)


class CueAdapter(nn.Module):
    """
    Maps cue prompt embeddings to M cue tokens.

    Args:
        input_dim (int): Input dimension (text encoder output)
        output_dim (int): Output token dimension
        n_tokens (int): Number of output tokens (M)
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 768, n_tokens: int = 8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.adaptive_pool = nn.AdaptiveMaxPool1d(n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert cue prompts to cue tokens."""
        # x: (batch, seq_len, input_dim)
        x = self.layers(x)  # (batch, seq_len, output_dim)
        x = x.transpose(1, 2)  # (batch, output_dim, seq_len)
        x = self.adaptive_pool(x)  # (batch, output_dim, n_tokens)
        return x.transpose(1, 2)  # (batch, n_tokens, output_dim)


class AttentionPool(nn.Module):
    """
    Attention-based pooling to get utterance-level summaries.

    Args:
        dim (int): Feature dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) / (dim ** 0.5))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool tokens to get summary: (batch, seq_len, dim) -> (batch, dim)"""
        # x: (batch, seq_len, dim)
        attn = self.softmax(torch.matmul(x, self.query.transpose(1, 2)))  # (batch, seq_len, 1)
        return (attn * x).sum(dim=1)  # (batch, dim)


class SCAF(nn.Module):
    """
    Speech-Cue Agreement Fusion (SCAF) module.

    Weights each cue by how reliable the cue stream looks and how well
    it agrees with the speech representation.

    Args:
        speech_dim (int): Dimension of speech tokens
        cue_dim (int): Dimension of cue tokens
        use_reliability (bool): Include reliability scoring
        use_agreement (bool): Include token agreement
        use_gate (bool): Use gated fusion
    """

    def __init__(
        self,
        speech_dim: int = 768,
        cue_dim: int = 768,
        use_reliability: bool = True,
        use_agreement: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_reliability = use_reliability
        self.use_agreement = use_agreement
        self.use_gate = use_gate

        # Attention pooling
        self.speech_pool = AttentionPool(speech_dim)
        self.cue_pool = AttentionPool(cue_dim)

        # Reliability scoring
        if use_reliability:
            self.reliability_scorer = nn.Sequential(
                nn.Linear(speech_dim + cue_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        # Agreement scoring
        if use_agreement:
            self.agreement_scorer = nn.Sequential(
                nn.Linear(speech_dim + cue_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cue_dim,
            num_heads=4,
            batch_first=True,
        )

        # Gated fusion
        if use_gate:
            self.gate_fusion = nn.Sequential(
                nn.Linear(speech_dim + cue_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

    def forward(self, speech_tokens: torch.Tensor, cue_tokens: torch.Tensor) -> torch.Tensor:
        """
        Fuse speech and cue streams with agreement weighting.

        Args:
            speech_tokens: (batch, n_speech_tokens, speech_dim)
            cue_tokens: (batch, n_cue_tokens, cue_dim)

        Returns:
            Fused features: (batch, n_speech_tokens + n_cue_tokens, cue_dim)
        """
        batch_size = speech_tokens.shape[0]

        # Get summaries
        speech_summary = self.speech_pool(speech_tokens)  # (batch, speech_dim)
        cue_summary = self.cue_pool(cue_tokens)  # (batch, cue_dim)

        # Compute reliability
        if self.use_reliability:
            combined = torch.cat([speech_summary, cue_summary], dim=1)
            reliability = self.reliability_scorer(combined)  # (batch, 1)
        else:
            reliability = 1.0

        # Compute agreement
        if self.use_agreement:
            combined = torch.cat([speech_summary, cue_summary], dim=1)
            agreement = self.agreement_scorer(combined)  # (batch, 1)
        else:
            agreement = 1.0

        # Cross-attention
        cross_attn_out, _ = self.cross_attention(
            speech_tokens, cue_tokens, cue_tokens
        )

        # Weight cue tokens
        weighted_cue = cue_tokens * reliability * agreement

        # Gated fusion
        if self.use_gate:
            combined = torch.cat([speech_tokens.mean(dim=1), weighted_cue.mean(dim=1)], dim=1)
            gate = self.gate_fusion(combined)  # (batch, 1)
            fused = gate * speech_tokens + (1 - gate) * weighted_cue
        else:
            fused = torch.cat([speech_tokens, weighted_cue], dim=1)

        return fused
