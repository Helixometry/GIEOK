"""Clinical cue measurement and processing for Alzheimer's detection."""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CueMarkers:
    """Clinical cue markers for Alzheimer's detection."""

    # Timing markers
    pause_duration: Optional[float] = None
    pause_count: Optional[int] = None
    speech_rate: Optional[float] = None

    # Fluency markers
    filled_pauses: Optional[int] = None
    repetitions: Optional[int] = None
    revisions: Optional[int] = None

    # ASR markers
    asr_confidence: Optional[float] = None
    unknown_words: Optional[int] = None

    # Prosodic markers
    mean_pitch: Optional[float] = None
    pitch_variance: Optional[float] = None
    energy_mean: Optional[float] = None


class CueProcessor:
    """
    Process and discretise clinical cues.

    Clinical cues are discretised into Low/Medium/High using 33rd and 66th
    percentiles computed from the English training split only.
    """

    def __init__(self):
        self.thresholds = {}

    def fit_thresholds(self, cue_values: Dict[str, List[float]]) -> None:
        """
        Fit percentile thresholds from training data.

        Args:
            cue_values: Dictionary mapping cue names to lists of values
        """
        for cue_name, values in cue_values.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                p33 = np.percentile(valid_values, 33)
                p66 = np.percentile(valid_values, 66)
                self.thresholds[cue_name] = {"p33": p33, "p66": p66}

    def discretise(self, cue_name: str, value: float) -> str:
        """
        Discretise a cue value into Low/Medium/High.

        Args:
            cue_name: Name of the cue
            value: Cue value to discretise

        Returns:
            Discretised level: "Low", "Medium", or "High"
        """
        if cue_name not in self.thresholds:
            return "Unknown"

        p33 = self.thresholds[cue_name]["p33"]
        p66 = self.thresholds[cue_name]["p66"]

        if value <= p33:
            return "Low"
        elif value <= p66:
            return "Medium"
        else:
            return "High"

    def create_prompt(self, markers: CueMarkers) -> str:
        """
        Create a clinical evidence prompt from cue markers.

        Args:
            markers: CueMarkers object with measured values

        Returns:
            Textual cue prompt for language model
        """
        cues_text = []

        # Timing
        if markers.pause_duration is not None:
            level = self.discretise("pause_duration", markers.pause_duration)
            cues_text.append(f"Pause duration: {level}")

        if markers.speech_rate is not None:
            level = self.discretise("speech_rate", markers.speech_rate)
            cues_text.append(f"Speech rate: {level}")

        # Fluency
        if markers.filled_pauses is not None:
            level = self.discretise("filled_pauses", markers.filled_pauses)
            cues_text.append(f"Filled pauses: {level}")

        if markers.repetitions is not None:
            level = self.discretise("repetitions", markers.repetitions)
            cues_text.append(f"Repetitions: {level}")

        # ASR
        if markers.asr_confidence is not None:
            level = self.discretise("asr_confidence", markers.asr_confidence)
            cues_text.append(f"ASR confidence: {level}")

        # Prosody
        if markers.mean_pitch is not None:
            level = self.discretise("mean_pitch", markers.mean_pitch)
            cues_text.append(f"Pitch: {level}")

        if not cues_text:
            return "No clinical cues measured."

        prompt = "Clinical evidence: " + "; ".join(cues_text) + "."
        return prompt
