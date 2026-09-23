"""Unit tests for the GIEOK components."""
import numpy as np
import pandas as pd
import pytest
import torch

from gieok.cues import CUE_SPEC, CueSet, measure, timing_cues, transcript_cues
from gieok.data import AdDataset, CueBank, cue_prompts, fit_cue_thresholds, fix_length, load_manifest, \
    make_synthetic, speaker_split
from gieok.metrics import mcnemar, scores
from gieok.model import ANSWERS, DIAGNOSTIC_PROMPT, Gieok, GieokConfig, SpeechProbe, cue_vocabulary
from gieok.modules import SCAF, AttentionPool, CueAdapter, TemporalAdapter, agreement_loss, reliability_loss

TOY = {"backbone": "toy", "toy_hidden": 32}


def small_config(**kw) -> GieokConfig:
    kw.setdefault("lm", TOY)
    return GieokConfig(speech_dim=16, dim=24, speech_tokens=6, cue_tokens=4, prefix_tokens=3,
                       heads=2, **kw)


def batch(b=4, frames=10, dim=16, cue_tokens=7, hidden=32):
    return {"features": torch.randn(b, frames, dim),
            "cue_hidden": torch.randn(b, cue_tokens, hidden),
            "cue_mask": torch.ones(b, cue_tokens),
            "label": torch.tensor([0, 1] * (b // 2))}


# ---------------------------------------------------------------- cues
def test_cue_levels_follow_the_source_training_percentiles():
    values = pd.DataFrame({name: np.arange(100, dtype=float) for name in CUE_SPEC})
    cueset = CueSet.fit(values)
    t33, t66 = cueset.thresholds["pause_ratio"]
    assert 32 <= t33 <= 34 and 65 <= t66 <= 67
    assert cueset.level("pause_ratio", 5) == "low"
    assert cueset.level("pause_ratio", 50) == "medium"
    assert cueset.level("pause_ratio", 95) == "high"
    assert cueset.level("pause_ratio", float("nan")) == "unavailable"
    assert cueset.level("not_a_cue", 1.0) == "unavailable"


def test_verbalised_prompt_mentions_every_measured_cue_group():
    values = pd.DataFrame({name: np.arange(100, dtype=float) for name in CUE_SPEC})
    prompt = CueSet.fit(values).verbalise({name: 95.0 for name in CUE_SPEC})
    for group in ("timing", "fluency", "recognition", "prosody"):
        assert f"{group}:" in prompt
    assert "high pause ratio" in prompt and prompt.startswith("Clinical speech evidence. The speech shows")


def test_prompt_leaves_out_cues_that_were_not_measured():
    values = pd.DataFrame({name: np.arange(100, dtype=float) for name in CUE_SPEC})
    cueset = CueSet.fit(values)
    partial = {name: 95.0 for name in CUE_SPEC}
    for name in ("filled_pause_rate", "repetition_rate", "words_per_minute",
                 "asr_confidence_mean", "asr_low_confidence_rate"):
        partial[name] = float("nan")
    prompt = cueset.verbalise(partial)
    assert "fluency" not in prompt and "recognition" not in prompt and "timing:" in prompt


def test_prompt_without_any_cue_values_is_still_valid():
    prompt = CueSet().verbalise({name: float("nan") for name in CUE_SPEC})
    assert "No reliable cue measurements" in prompt


def test_cue_thresholds_roundtrip(tmp_path):
    cueset = CueSet.fit(pd.DataFrame({name: np.arange(10, dtype=float) for name in CUE_SPEC}))
    cueset.to_json(tmp_path / "t.json")
    assert CueSet.from_json(tmp_path / "t.json").thresholds == cueset.thresholds


def test_timing_cues_detect_a_long_pause():
    sr = 16_000
    speech = np.random.randn(sr).astype(np.float32) * 0.2
    silence = np.zeros(2 * sr, dtype=np.float32)
    quiet = timing_cues(np.concatenate([speech, silence, speech]), sr)
    busy = timing_cues(np.concatenate([speech, speech, speech]), sr)
    assert quiet["pause_ratio"] > busy["pause_ratio"]
    assert quiet["mean_pause_duration"] > busy["mean_pause_duration"]


def test_transcript_cues_count_filled_pauses_and_repetitions():
    words = [{"word": "the", "confidence": 0.9}, {"word": "uh", "confidence": 0.3},
             {"word": "cat", "confidence": 0.8}, {"word": "cat", "confidence": 0.2}]
    cues = transcript_cues(words, duration_s=60.0)
    assert cues["filled_pause_rate"] == pytest.approx(1.0)
    assert cues["repetition_rate"] == pytest.approx(1 / 3)
    assert cues["asr_low_confidence_rate"] == pytest.approx(0.5)


def test_measure_without_a_transcript_marks_those_cues_unavailable():
    wav = np.random.randn(16_000).astype(np.float32) * 0.1
    values = measure(wav)
    assert np.isfinite(values["pause_ratio"]) and np.isfinite(values["pitch_variability"])
    assert np.isnan(values["filled_pause_rate"]) and np.isnan(values["asr_confidence_mean"])


# ---------------------------------------------------------------- modules
def test_adapters_produce_the_requested_number_of_tokens():
    assert TemporalAdapter(16, 24, 6)(torch.randn(3, 40, 16)).shape == (3, 6, 24)
    assert CueAdapter(32, 24, 4)(torch.randn(3, 9, 32)).shape == (3, 4, 24)
    assert AttentionPool(24)(torch.randn(3, 6, 24)).shape == (3, 24)


def test_cue_adapter_ignores_padding():
    """A short prompt gives the same cue tokens however much padding follows it."""
    torch.manual_seed(0)
    adapter = CueAdapter(32, 24, 4).eval()
    hidden = torch.randn(1, 5, 32)
    short = adapter(hidden, torch.ones(1, 5))
    padded = adapter(torch.cat([hidden, torch.randn(1, 20, 32)], 1),
                     torch.cat([torch.ones(1, 5), torch.zeros(1, 20)], 1))
    assert torch.allclose(short, padded, atol=1e-5)


def test_scaf_outputs_are_bounded_and_speech_shaped():
    scaf = SCAF(24, heads=2)
    out = scaf(torch.randn(4, 6, 24), torch.randn(4, 5, 24))
    assert out["fused"].shape == (4, 6, 24)
    assert out["reliability"].shape == (4,) and ((0 < out["reliability"]) & (out["reliability"] < 1)).all()
    assert out["agreement"].shape == (4, 5) and ((0 < out["agreement"]) & (out["agreement"] < 1)).all()


def test_scaf_switches_change_the_fusion():
    torch.manual_seed(0)
    speech, cue = torch.randn(4, 6, 24), torch.randn(4, 5, 24)
    full = SCAF(24, heads=2)
    plain = SCAF(24, heads=2, use_reliability=False, use_agreement=False, use_gate=False)
    plain.load_state_dict(full.state_dict())
    full.eval(), plain.eval()
    assert not torch.allclose(full(speech, cue)["fused"], plain(speech, cue)["fused"])
    assert plain(speech, cue)["gate"].abs().sum() == 0        # no gate: the cross-attended output is used


def test_auxiliary_losses():
    assert reliability_loss(torch.full((8,), 0.5)) < reliability_loss(torch.full((8,), 0.01))
    z = torch.randn(8, 16)
    assert agreement_loss(z, z.clone()) < agreement_loss(z, torch.randn(8, 16))
    assert agreement_loss(torch.randn(1, 16), torch.randn(1, 16)) == 0


# ---------------------------------------------------------------- model
@pytest.mark.parametrize("variant", [{}, {"fusion": "crossattn"}, {"fusion": "concat"}, {"fusion": "none"},
                                     {"use_reliability": False}, {"use_agreement": False},
                                     {"use_gate": False}, {"lambda_agree": 0.0}])
def test_gieok_variants_forward_backward(variant):
    model = Gieok(small_config(**variant))
    data = batch()
    out = model(data)
    loss, parts = model.loss(out, data)
    loss.backward()
    assert out["logits"].shape == (4, 2) and "lm" in parts
    assert model.speech_adapter.conv[0].weight.grad is not None
    assert not any(p.requires_grad for p in model.lm.lm.parameters())      # decoder stays frozen


def test_answers_and_prompt_match_the_paper():
    assert ANSWERS == ("Healthy", "Alzheimer")
    assert DIAGNOSTIC_PROMPT.startswith("Determine whether the speech corresponds to Alzheimer")
    assert "Reply in one word" in DIAGNOSTIC_PROMPT
    assert any("pause ratio" in phrase for phrase in cue_vocabulary())


def test_speech_probe_baseline():
    model = SpeechProbe(small_config())
    data = batch()
    out = model(data)
    loss, _ = model.loss(out, data)
    loss.backward()
    assert out["logits"].shape == (4, 2)


# ---------------------------------------------------------------- data
def test_fix_length_crops_and_pads():
    assert fix_length(np.zeros((10, 4)), 6).shape == (6, 4)
    assert fix_length(np.zeros((3, 4)), 6).shape == (6, 4)
    assert fix_length(np.zeros(4), 2).shape == (2, 4)


def test_synthetic_corpus_and_pipeline(tmp_path):
    manifest = make_synthetic(tmp_path, dim=8, frames=6, speakers_per_language=6, per_speaker=2)
    df = load_manifest(manifest)
    assert set(df["language"]) == {"en", "zh", "es", "el"}
    assert set(df.loc[df["language"] != "en", "split"]) == {"test"}
    cueset = fit_cue_thresholds(df, "en")
    assert set(cueset.thresholds) == set(CUE_SPEC)
    prompts = cue_prompts(df, cueset)
    assert len(prompts) == len(df) and all(p.startswith("Clinical speech evidence.") for p in prompts)

    model = Gieok(GieokConfig(speech_dim=8, dim=24, speech_tokens=6, cue_tokens=4, prefix_tokens=3, heads=2, lm=TOY))
    bank = CueBank(prompts, model.lm)
    dataset = AdDataset(df, range(4), num_frames=6, bank=bank)
    item = dataset[0]
    assert item["features"].shape == (6, 8) and item["cue_hidden"].shape[-1] == model.lm.hidden_size


def test_speaker_split_is_disjoint():
    df = pd.DataFrame({"speaker_id": [f"S{i // 3}" for i in range(30)]})
    large, small = speaker_split(df, range(30), frac=0.25, seed=0)
    assert set(large) & set(small) == set()
    assert set(df.iloc[large]["speaker_id"]) & set(df.iloc[small]["speaker_id"]) == set()


def test_cue_thresholds_use_only_source_training_rows():
    df = pd.DataFrame({name: [1.0, 2.0, 3.0, 1000.0] for name in CUE_SPEC})
    df["language"], df["split"] = ["en", "en", "en", "zh"], ["train", "train", "train", "test"]
    thresholds = fit_cue_thresholds(df, "en").thresholds["pause_ratio"]
    assert max(thresholds) < 10          # the Chinese outlier never reaches the thresholds


# ---------------------------------------------------------------- metrics
def test_scores_and_mcnemar():
    y = np.array([0, 1, 0, 1])
    perfect = scores(y, y)
    assert perfect["acc"] == 100.0 and perfect["ad_f1"] == 100.0 and perfect["hc_acc"] == 100.0
    result = mcnemar(y, y, np.array([0, 0, 0, 0]))
    assert result["only_a_correct"] == 2 and result["only_b_correct"] == 0
    assert mcnemar(y, y, y)["p_value"] == 1.0


def test_gieok_with_a_hugging_face_decoder(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tests.tiny_hf import make_tiny_qwen2
    texts = [DIAGNOSTIC_PROMPT, " ".join(ANSWERS), *cue_vocabulary(), "low medium high"]
    model_dir, tok_dir = make_tiny_qwen2(str(tmp_path), texts)
    model = Gieok(small_config(lm={"backbone": model_dir, "tokenizer": tok_dir, "lora_rank": 2}))
    data = batch(hidden=model.lm.hidden_size)
    loss, _ = model.loss(model(data), data)
    loss.backward()
    assert torch.isfinite(loss)
