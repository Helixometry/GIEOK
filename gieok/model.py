"""Core GIEOK model for zero-shot cross-lingual Alzheimer's detection."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class GIEOK(nn.Module):
    """
    GIEOK: Paralinguistic Audio-Language Model for Alzheimer's Detection.

    Detects Alzheimer's disease in zero-shot cross-lingual settings by combining
    speech and clinical cues through Speech-Cue Agreement Fusion (SCAF).

    Args:
        speech_dim (int): Dimension of speech encoder output (1024 for TRILLsson)
        cue_dim (int): Dimension of cue embeddings
        decoder_dim (int): Dimension of language model embeddings (4096 for Qwen2.5)
        n_speech_tokens (int): Number of speech tokens (K)
        n_cue_tokens (int): Number of cue tokens (M)
        use_scaf (bool): Enable Speech-Cue Agreement Fusion
        use_reliability (bool): Use reliability scoring in SCAF
        use_agreement (bool): Use token agreement in SCAF
        use_gate (bool): Use gated fusion in SCAF
    """

    def __init__(
        self,
        speech_dim: int = 1024,
        cue_dim: int = 768,
        decoder_dim: int = 4096,
        n_speech_tokens: int = 16,
        n_cue_tokens: int = 8,
        use_scaf: bool = True,
        use_reliability: bool = True,
        use_agreement: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.speech_dim = speech_dim
        self.cue_dim = cue_dim
        self.decoder_dim = decoder_dim
        self.use_scaf = use_scaf

        # Temporal adapter: maps speech features to speech tokens
        self.temporal_adapter = nn.Sequential(
            nn.Conv1d(speech_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, cue_dim, kernel_size=3, padding=1),
            nn.AdaptiveMaxPool1d(n_speech_tokens),
        )

        # Cue adapter: maps cue prompts to cue tokens
        self.cue_adapter = nn.Sequential(
            nn.Linear(cue_dim, 512),
            nn.ReLU(),
            nn.Linear(512, cue_dim),
        )

        # SCAF components (if enabled)
        if use_scaf:
            from .modules import SCAF
            self.scaf = SCAF(
                speech_dim=cue_dim,
                cue_dim=cue_dim,
                use_reliability=use_reliability,
                use_agreement=use_agreement,
                use_gate=use_gate,
            )

        # Prefix projection: maps fused features to decoder embedding space
        self.prefix_projection = nn.Sequential(
            nn.Linear(cue_dim * (n_speech_tokens + n_cue_tokens), decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

        # Classification head for non-LLM setup
        self.classify_head = nn.Sequential(
            nn.Linear(decoder_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),  # Binary: Healthy vs Alzheimer
        )

    def forward(
        self,
        speech_features: torch.Tensor,
        cue_embeddings: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GIEOK.

        Args:
            speech_features: Speech encoder outputs (batch, seq_len, speech_dim)
            cue_embeddings: Cue prompt embeddings (batch, cue_seq_len, cue_dim)
            return_features: Return intermediate features

        Returns:
            Dictionary with logits and optional intermediate features
        """
        # Temporal adaptation: speech -> speech tokens
        batch_size = speech_features.shape[0]
        speech_tokens = self.temporal_adapter(speech_features.transpose(1, 2))
        speech_tokens = speech_tokens.transpose(1, 2)  # (batch, n_tokens, cue_dim)

        # Cue adaptation: cue prompts -> cue tokens
        cue_tokens = self.cue_adapter(cue_embeddings)  # (batch, n_cue_tokens, cue_dim)

        # SCAF fusion
        if self.use_scaf:
            fused_features = self.scaf(speech_tokens, cue_tokens)
        else:
            # Simple concatenation
            fused_features = torch.cat([speech_tokens, cue_tokens], dim=1)

        # Flatten for classification
        fused_features_flat = fused_features.reshape(batch_size, -1)

        # Project to decoder space / classification
        logits = self.prefix_projection(fused_features_flat)

        return {
            "logits": logits,
            "speech_tokens": speech_tokens if return_features else None,
            "cue_tokens": cue_tokens if return_features else None,
            "fused": fused_features if return_features else None,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "GIEOK":
        """
        Load a pretrained GIEOK model.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            GIEOK model with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """
        Save the model checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint
            config: Configuration dictionary
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "speech_dim": self.speech_dim,
                "cue_dim": self.cue_dim,
                "decoder_dim": self.decoder_dim,
            },
        }
        torch.save(checkpoint, checkpoint_path)
