"""Documentation for GIEOK."""

# GIEOK Architecture

## Overview

GIEOK (Paralinguistic Audio-Language Model for Alzheimer's Detection) detects Alzheimer's disease in zero-shot cross-lingual settings by combining speech representations with clinical cues through **Speech-Cue Agreement Fusion (SCAF)**.

## Core Components

### 1. Speech Stream
- **Encoder**: Frozen TRILLsson (1024-dim) or Whisper-base (512-dim)
- **Adapter**: Maps to **K=16 speech tokens** using Conv1d + adaptive pooling
- Preserves temporal and prosodic structure

### 2. Cue Stream
- **Clinical Markers**: 12 markers in 4 groups
  - Timing: pause duration, pause count, speech rate
  - Fluency: filled pauses, repetitions, revisions
  - ASR: ASR confidence, unknown words
  - Prosody: pitch, energy
- **Discretisation**: Low/Medium/High using 33rd/66th percentiles (English training only)
- **Encoding**: Frozen text pathway from Qwen2.5-7B
- **Adapter**: Maps cue prompts to **M=8 cue tokens**

### 3. Speech-Cue Agreement Fusion (SCAF)

```
Speech tokens (Z_s) ────┐
                        ├─ Attention pooling ─┐
                        │                     ├─ Reliability score (r)
Cue tokens (Z_c) ──────┤                     │
                        └─ Attention pooling ─┤
                                            │
                                      Token agreement (a_m)
                                            │
                        Weighted cue: z̃_c,m = r · a_m · z_c,m
                        
                        Cross-attention: U = CrossAttn(Z_s, Z̃_c)
                        
                        Gated fusion: Z_f = β ⊙ Z_s + (1 - β) ⊙ U
```

### 4. Language Model Decoder
- Frozen Qwen2.5-7B
- Fused features mapped to prefix tokens
- Constrained decoding: only "Healthy" or "Alzheimer"

## Training Objective

```
L_total = L_LM + λ_rel · L_rel + λ_agree · L_agree

L_LM: Language modeling loss on binary choice
L_rel: Reliability regularization - prevents collapse
L_agree: InfoNCE loss - pulls matched speech/cue summaries together
```

## Zero-Shot Cross-Lingual Transfer

- **Train on**: English speech (309 AD, 243 HC)
- **Test on**: Chinese, Spanish, Greek (zero-shot, no adaptation)
- **Cue thresholds**: Fixed from English training, reused unchanged

### Key Insight
Clinical cues carry diagnostic evidence, but not equally in every recording. Pause patterns reflect language speaking style; low ASR confidence can mean accent. SCAF weights each cue by reliability for that utterance.

## Results Summary

| Metric | En | Zh | Es | El | Avg (Zh/Es/El) |
|--------|----|----|----|----|-----------------|
| **Accuracy** | 94.63% | 83.74% | 80.86% | 77.95% | 80.85% |
| **F1 (macro)** | 92.18% | 82.29% | 79.41% | 74.32% | 78.67% |

SCAF outperforms baselines by 5-7% on zero-shot languages.

## File Organization

- `model.py`: Core GIEOK architecture
- `modules.py`: SCAF, adapters, attention pooling
- `cues.py`: Cue measurement and discretisation
- `features.py`: TRILLsson/Whisper feature extraction
- `data.py`: Manifest loading, speaker-independent splits
- `train.py`: Training loop
- `compare.py`: McNemar's test for result comparison
- `metrics.py`: Accuracy, F1, class-wise scores

## Training Configuration

- Encoder: frozen TRILLsson (1024-d)
- Temporal adapter: Conv1d → 16 speech tokens
- Cue adapter: linear → 8 cue tokens
- SCAF: 4-head cross-attention
- Prefix: 8 tokens → decoder embedding space
- Optimizer: AdamW (lr=1e-4, wd=1e-2)
- Batch: 8, Epochs: 50
- Loss: L_LM + λ_rel·L_rel + λ_agree·L_agree

## Ablation Study

| Variant | En | Avg (Zh/Es/El) |
|---------|----|----|
| No cue prompt | 82.15% | 70.24% |
| Cue + concatenation | 88.37% | 74.87% |
| SCAF w/o reliability | 91.70% | 78.59% |
| SCAF w/o agreement | 89.00% | 71.31% |
| **GIEOK (full SCAF)** | **94.63%** | **80.85%** |

Removing token-level agreement hurts most (~9% drop on En).
