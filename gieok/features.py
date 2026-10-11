"""Speech feature extraction for GIEOK."""

import numpy as np
import librosa
from pathlib import Path
from typing import Union, Optional
import torch


class FeatureExtractor:
    """
    Extract speech features using TRILLsson or Whisper encoders.

    Args:
        model_name (str): 'trillsson' or 'whisper'
        sample_rate (int): Sample rate for audio
    """

    def __init__(self, model_name: str = "trillsson", sample_rate: int = 16000):
        self.model_name = model_name
        self.sample_rate = sample_rate

        if model_name == "trillsson":
            try:
                # TRILLsson: freezes at 1024-dim features
                self.encoder = self._load_trillsson()
                self.output_dim = 1024
            except Exception as e:
                raise RuntimeError(f"Failed to load TRILLsson: {e}")

        elif model_name == "whisper":
            try:
                import whisper
                self.encoder = whisper.load_model("base")
                self.output_dim = 512
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper: {e}")

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _load_trillsson(self):
        """Load TRILLsson encoder."""
        try:
            import tensorflow_hub as hub
            # TRILLsson model from TensorFlow Hub
            return hub.load('https://tfhub.dev/google/trillsson/1')
        except:
            # Fallback: create a placeholder
            return None

    def extract(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Extract frame-level features from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Feature array (seq_len, output_dim)
        """
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)

        if self.model_name == "trillsson":
            return self._extract_trillsson(audio)
        elif self.model_name == "whisper":
            return self._extract_whisper(audio)

    def _extract_trillsson(self, audio: np.ndarray) -> np.ndarray:
        """Extract TRILLsson features."""
        # TRILLsson outputs at 20ms frames
        frame_size = int(0.02 * self.sample_rate)
        hop_size = int(0.01 * self.sample_rate)

        features = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            # Placeholder: in real implementation, use model
            features.append(np.random.randn(1024))

        return np.array(features)

    def _extract_whisper(self, audio: np.ndarray) -> np.ndarray:
        """Extract Whisper encoder features."""
        # Whisper processes audio in 30-second chunks
        chunk_size = int(30 * self.sample_rate)
        features = []

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            # Placeholder: use actual Whisper encoder
            features.append(np.random.randn(512))

        return np.array(features)
