"""Tests for the S2 cross-encoder teacher + distillation (B-21 W2/W3).

The load-bearing property is the distillation target itself: ``blend·one_hot +
(1-blend)·softmax(logits/T)`` is a proper distribution anchored on the gold option
(ADR-0005), and the NLI rows of the arm are recognised by their own text shape. The GPU
training path is exercised by the cycle, not here (the ``train`` extra stays lazy).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.train_cross_encoder import (
    NLI_LABELS,
    TeacherConfig,
    criterion_texts_for,
    is_nli_record,
    load_config,
    nli_class_index,
    soft_targets_from_logits,
    split_nli,
)

CONFIG = Path(__file__).resolve().parent.parent / "training" / "configs" / "cross_encoder_s2.json"


def test_config_loads_and_is_strict() -> None:
    config = load_config(CONFIG)
    assert config.epochs == 2
    assert config.seed == 2
    assert config.nli_weight == 0.5
    with pytest.raises(ValueError, match="unknown config keys"):
        TeacherConfig.from_dict({"epochs": 1, "surprise": True})


def test_split_nli_reads_the_premise_hypothesis_pair() -> None:
    pair = split_nli("Premise: the seals are intact.\nHypothesis: the parcel is unopened.")
    assert pair == ("the seals are intact.", "the parcel is unopened.")
    assert split_nli("A helpdesk classifier must decide.") is None
    assert split_nli("Premise: only a premise") is None


def test_is_nli_record_recognises_the_arms_nli_rows() -> None:
    record = {
        "type": "choice",
        "state": "Premise: a.\nHypothesis: b.",
        "criteria": {label: label for label in NLI_LABELS},
        "target": "neutral",
    }
    assert is_nli_record(record)
    assert nli_class_index(record) == NLI_LABELS.index("neutral")
    assert not is_nli_record({**record, "target": "neutral", "state": "no pair here"})
    assert not is_nli_record({**record, "criteria": {"approve": "yes", "deny": "no"}})


def test_soft_targets_are_a_proper_distribution_anchored_on_gold() -> None:
    target = soft_targets_from_logits([0.0, 4.0, -1.0], target_index=0, temperature=2.0, blend=0.5)
    assert len(target) == 3
    assert sum(target) == pytest.approx(1.0)
    assert min(target) >= 0.0
    # the hard-label anchor keeps at least the blend mass on the gold option
    assert target[0] >= 0.5
    # the teacher's shape survives the flattening: the strongest logit stays strongest
    assert target[1] > target[2]


def test_soft_targets_refuse_bad_shapes() -> None:
    with pytest.raises(ValueError, match="temperature"):
        soft_targets_from_logits([1.0, 2.0], target_index=0, temperature=0.0)
    with pytest.raises(ValueError, match="blend"):
        soft_targets_from_logits([1.0, 2.0], target_index=0, blend=1.5)
    with pytest.raises(ValueError, match="target_index"):
        soft_targets_from_logits([1.0, 2.0], target_index=5)
    with pytest.raises(ValueError, match="non-empty"):
        soft_targets_from_logits([], target_index=0)


def test_teacher_config_round_trips(tmp_path: Path) -> None:
    config = TeacherConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_config(path) == config


def test_criterion_presentations_cover_the_measured_difference() -> None:
    record = {"criteria": {"math_agent": "Numbers.", "coding_agent": ""}}
    assert criterion_texts_for(record, "training") == ["math_agent: Numbers.", "coding_agent"]
    assert criterion_texts_for(record, "bare") == ["Numbers.", "coding_agent"]
    with pytest.raises(ValueError, match="presentation"):
        criterion_texts_for(record, "loud")
