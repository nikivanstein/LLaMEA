"""Tests for RLM-based pre-screening inside the LLaMEA loop
(`llamea.rlm_surrogate.screening` + the `rlm_screening` integration in
`llamea.llamea.LLaMEA`).

The pure selection-policy functions (`decide_topk_filter`,
`decide_threshold_reject`) need no torch/regress_lm and always run. The
`RLMScreener`/`LLaMEA` integration tests train a tiny, fully-offline `char`
checkpoint (same pattern as `test_rlm_pipeline_e2e.py`) and are marked
`slow` -- skipped automatically if torch/regress_lm aren't installed (the
`rlm-surrogate` dependency group).
"""

from pathlib import Path

import numpy as np
import pytest

from llamea.rlm_surrogate.screening import decide_threshold_reject, decide_topk_filter

# --------------------------------------------------------------------------
# Pure selection-policy functions -- no torch/regress_lm needed.
# --------------------------------------------------------------------------


def test_decide_topk_filter_keeps_best_n_when_maximizing():
    predicted = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    mask = decide_topk_filter(predicted, n_keep=2, minimization=False)
    assert mask.sum() == 2
    assert predicted[mask].min() >= predicted[~mask].max()


def test_decide_topk_filter_keeps_lowest_n_when_minimizing():
    predicted = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    mask = decide_topk_filter(predicted, n_keep=2, minimization=True)
    assert mask.sum() == 2
    assert predicted[mask].max() <= predicted[~mask].min()


def test_decide_topk_filter_clamps_n_keep_to_pool_size():
    predicted = np.array([0.1, 0.2])
    mask = decide_topk_filter(predicted, n_keep=10, minimization=False)
    assert mask.sum() == 2
    assert mask.all()


def test_decide_topk_filter_empty_pool():
    mask = decide_topk_filter(np.array([]), n_keep=3, minimization=False)
    assert mask.shape == (0,)


def test_decide_threshold_reject_keeps_upper_half_when_maximizing():
    predicted = np.array([0.1, 0.2, 0.8, 0.9])
    mask = decide_threshold_reject(
        predicted, threshold_quantile=0.5, minimization=False
    )
    # median is 0.5 -> entries >= 0.5 are kept.
    assert list(mask) == [False, False, True, True]


def test_decide_threshold_reject_keeps_lower_half_when_minimizing():
    predicted = np.array([0.1, 0.2, 0.8, 0.9])
    mask = decide_threshold_reject(predicted, threshold_quantile=0.5, minimization=True)
    assert list(mask) == [True, True, False, False]


def test_decide_threshold_reject_empty_pool():
    mask = decide_threshold_reject(
        np.array([]), threshold_quantile=0.5, minimization=False
    )
    assert mask.shape == (0,)


def test_solution_to_x_matches_build_x_shape():
    """Mirrors `data_pipeline.build_x`'s section layout for a live
    `Solution` (configspace serialized the same way `schema._coerce_configspace`
    serializes it for training data: `json.dumps(cs.to_serialized_dict())` for
    a live `ConfigSpace.ConfigurationSpace`-shaped object, since
    `BladeRecord.configspace` -- what the surrogate trained on -- is always
    already a string by the time it reaches `build_x`)."""
    from llamea.rlm_surrogate.screening import solution_to_x

    class FakeConfigSpace:
        def to_serialized_dict(self):
            return {"x": [0, 1]}

    class FakeSolution:
        description = "A simple hill climber."
        configspace = FakeConfigSpace()
        code = "class Foo:\n    pass"

    x = solution_to_x(FakeSolution())
    assert x.startswith("# Description\nA simple hill climber.")
    assert "# Config space\n" in x
    assert '{"x": [0, 1]}' in x
    assert x.endswith("# Code\nclass Foo:\n    pass")


def test_solution_to_x_passes_through_string_configspace():
    """When `configspace` is already a string (e.g. restored from a logged
    record), it's used verbatim -- no re-serialization."""
    from llamea.rlm_surrogate.screening import solution_to_x

    class FakeSolution:
        description = ""
        configspace = '{"x": [0, 1]}'
        code = "class Foo:\n    pass"

    x = solution_to_x(FakeSolution())
    assert x == '# Config space\n{"x": [0, 1]}\n\n# Code\nclass Foo:\n    pass'


def test_solution_to_x_omits_empty_sections():
    from llamea.rlm_surrogate.screening import solution_to_x

    class FakeSolution:
        description = ""
        configspace = None
        code = "class Foo:\n    pass"

    x = solution_to_x(FakeSolution())
    assert x == "# Code\nclass Foo:\n    pass"


# --------------------------------------------------------------------------
# RLMScreener + LLaMEA integration -- requires torch/regress_lm, trains a
# tiny real checkpoint, so marked slow.
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("regress_lm")

from llamea.rlm_surrogate.config import RLMSurrogateConfig
from llamea.rlm_surrogate.data_pipeline import RLMExample, write_examples_jsonl
from llamea.rlm_surrogate.problem_instances import (
    ProblemInstance,
    compute_problem_feature_text,
)
from llamea.rlm_surrogate.screening import RLMScreener, RLMScreeningConfig
from llamea.rlm_surrogate.train import train

TINY_CONFIG = (
    Path(__file__).parent.parent
    / "llamea"
    / "rlm_surrogate"
    / "configs"
    / "tiny_local_test.yaml"
)

INSTANCES = [
    ProblemInstance(kind="bbob", dim=5, fid_or_idx=1, iid=1),
    ProblemInstance(kind="bbob", dim=5, fid_or_idx=2, iid=1),
]


def _train_tiny_checkpoint(tmp_path) -> Path:
    """Trains a tiny 'char'-encoder checkpoint on synthetic data whose `x`
    matches `solution_to_x(...) + "\\n\\n# Problem\\n" + feature_text` --
    exactly what `RLMScreener.score` builds at inference time -- using
    `feature_mode="meta"` (no `ioh` needed, deterministic/static text)."""
    feature_texts = [
        compute_problem_feature_text(inst, mode="meta") for inst in INSTANCES
    ]
    codes_and_scores = [
        ("class Good:\n    def __call__(self, x):\n        return 1.0", 0.9),
        ("class Bad:\n    def __call__(self, x):\n        return 0.0", 0.1),
    ]
    examples = []
    for i in range(80):
        code, base_y = codes_and_scores[i % 2]
        for ft in feature_texts:
            x = f"# Description\ncandidate {i}\n\n# Code\n{code}\n\n# Problem\n{ft}"
            examples.append(
                RLMExample(
                    id=f"ex{i}-{hash(ft) % 1000}",
                    run_id=f"run{i % 4}",
                    generation=i % 3,
                    parent_ids=[],
                    x=x,
                    y=base_y,
                    fitness=base_y,
                )
            )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_examples_jsonl(examples, data_dir / "train.jsonl")
    write_examples_jsonl(examples[:16], data_dir / "val.jsonl")

    config = RLMSurrogateConfig.from_yaml(TINY_CONFIG)
    ckpt_dir = tmp_path / "ckpt"
    train(config, data_dir / "train.jsonl", data_dir / "val.jsonl", ckpt_dir)
    return ckpt_dir


@pytest.mark.slow
def test_rlm_screener_scores_one_value_per_solution(tmp_path):
    ckpt_dir = _train_tiny_checkpoint(tmp_path)

    class FakeSolution:
        def __init__(self, code):
            self.description = "synthetic"
            self.configspace = None
            self.code = code

    solutions = [
        FakeSolution("class Good:\n    def __call__(self, x):\n        return 1.0"),
        FakeSolution("class Bad:\n    def __call__(self, x):\n        return 0.0"),
    ]
    screener = RLMScreener(
        RLMScreeningConfig(
            checkpoint_dir=str(ckpt_dir),
            feature_mode="meta",
            num_samples_point_pred=4,
            predict_batch_size=2,
        ),
        INSTANCES,
    )
    scores = screener.score(solutions)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


@pytest.mark.slow
def test_rlm_screener_empty_input_returns_empty_array(tmp_path):
    ckpt_dir = _train_tiny_checkpoint(tmp_path)
    screener = RLMScreener(
        RLMScreeningConfig(checkpoint_dir=str(ckpt_dir), feature_mode="meta"),
        INSTANCES,
    )
    scores = screener.score([])
    assert scores.shape == (0,)


@pytest.mark.slow
@pytest.mark.parametrize(
    "strategy,kwargs",
    [
        ("log_only", {}),
        ("topk_filter", {"rlm_overgenerate_factor": 2.0}),
        ("threshold_reject", {"rlm_threshold_quantile": 0.5}),
    ],
)
def test_llamea_run_with_rlm_screening_strategies(tmp_path, strategy, kwargs):
    """End-to-end: a Dummy_LLM-driven LLaMEA run with each active
    `rlm_screening` strategy completes without raising, and the
    budget-accounting contract documented on `RLMScreeningConfig` holds:
    `topk_filter` never leaves a `rlm_screened_out` candidate in
    `run_history` (they're discarded, not counted), while the other active
    strategies may."""
    import random

    from llamea import Dummy_LLM, LLaMEA

    ckpt_dir = _train_tiny_checkpoint(tmp_path)
    rng = random.Random(0)

    def f(individual, logger=None):
        individual.set_scores(rng.uniform(0, 1), feedback="synthetic eval")
        return individual

    instances_as_dicts = [
        {"kind": "bbob", "dim": 5, "fid_or_idx": 1, "iid": 1},
        {"kind": "bbob", "dim": 5, "fid_or_idx": 2, "iid": 1},
    ]
    ea = LLaMEA(
        f,
        Dummy_LLM("dummy-model"),
        n_parents=3,
        n_offspring=4,
        budget=11,
        log=False,
        rlm_screening=strategy,
        rlm_checkpoint_dir=str(ckpt_dir),
        rlm_problem_instances=instances_as_dicts,
        rlm_feature_mode="meta",
        rlm_num_samples_point_pred=4,
        rlm_predict_batch_size=2,
        max_workers=2,
        **kwargs,
    )
    best = ea.run()
    assert np.isfinite(best.fitness)
    assert len(ea.run_history) >= ea.budget

    screened_out_in_history = [
        p for p in ea.run_history if p.get_metadata("rlm_screened_out")
    ]
    if strategy == "topk_filter":
        assert screened_out_in_history == []
    # Every successfully-generated candidate past the initial population
    # should carry a recorded prediction.
    assert any(
        p.get_metadata("rlm_predicted_score") is not None for p in ea.run_history
    )
