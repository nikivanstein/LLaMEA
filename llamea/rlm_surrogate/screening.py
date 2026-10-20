"""RLM-based pre-screening of freshly-sampled candidates inside the LLaMEA
evolutionary loop, so the (typically much more expensive) real benchmark
evaluation (``LLaMEA.f``) is only spent on candidates the trained surrogate
(see ``train.py``/``evaluate.py``) predicts are worth it.

Kept independent of ``llamea.py`` apart from the integration points it
calls into (``RLMScreener.score``) -- torch/regress_lm/ioh are all imported
lazily so importing ``llamea`` (or this module, until ``RLMScreener`` is
actually constructed) never requires them.

IMPORTANT -- inference/training config must match: ``feature_mode`` and
``n_lhs_points`` here are NOT read from the checkpoint (``RLMSurrogateConfig``
has no such fields -- they're a ``data_pipeline.py``-level choice, made when
the training examples were built). Passing a ``feature_mode``/``n_lhs_points``
that doesn't match what the checkpoint was actually trained on silently
produces out-of-distribution inputs and meaningless scores -- there is no
way to validate this from the checkpoint directory alone. Record and reuse
the exact `data_pipeline.py --feature-mode --lhs-points` values a checkpoint
was trained with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .problem_instances import ProblemInstance, compute_problem_feature_text

ScreeningStrategy = Literal["off", "log_only", "topk_filter", "threshold_reject"]

# Strategies that actually require a loaded RLMScreener (i.e. everything
# except the no-op "off").
ACTIVE_STRATEGIES = ("log_only", "topk_filter", "threshold_reject")


def solution_to_x(
    solution: Any,
    *,
    include_description: bool = True,
    include_configspace: bool = True,
) -> str:
    """Builds the RLM input text for a live ``llamea.solution.Solution``.

    Mirrors ``data_pipeline.build_x`` (description + configspace + code)
    exactly, including ``schema._coerce_configspace``'s
    ``json.dumps(...)`` serialization of a non-string configspace --
    ``Solution.configspace`` is a live ``ConfigSpace.ConfigurationSpace``
    object during the run, whereas ``BladeRecord.configspace`` (what the
    surrogate was trained on) is always a string, already serialized by the
    time it reaches ``build_x``. Matching that string format here is what
    keeps inference-time text in-distribution with training.
    """
    parts: list[str] = []
    description = (solution.description or "").strip()
    if include_description and description:
        parts.append(f"# Description\n{description}")
    if include_configspace and solution.configspace is not None:
        cs = solution.configspace
        if isinstance(cs, str):
            cs_text = cs
        else:
            try:
                cs_text = json.dumps(cs.to_serialized_dict())
            except Exception:
                cs_text = str(cs)
        cs_text = cs_text.strip()
        if cs_text:
            parts.append(f"# Config space\n{cs_text}")
    parts.append(f"# Code\n{solution.code}")
    return "\n\n".join(parts)


@dataclass
class RLMScreeningConfig:
    """Hyperparameters for RLM-based pre-screening in the LLaMEA loop.

    Attributes:
        checkpoint_dir: Directory written by ``train.py`` (``model.pt`` +
            ``config.yaml``).
        strategy: One of:
            - ``"off"``: screening disabled (the default, unused directly --
              ``LLaMEA`` only builds a screener when
              ``rlm_screening != "off"``).
            - ``"log_only"``: scores every offspring with the RLM and
              records the prediction in ``Solution.metadata["rlm_predicted_score"]``,
              but never skips real evaluation. A control condition for
              measuring prediction-vs-outcome correlation inside a live run
              without changing the run's trajectory.
            - ``"topk_filter"``: over-generates a pool of
              ``n_offspring * overgenerate_factor`` candidates per
              generation, scores all of them, and only sends the
              ``n_offspring`` most-promising ones to real evaluation. The
              rest are discarded (never added to ``run_history``, never
              counted against the evaluation budget) -- this is the
              "practical pre-screening" mode: more at-bats per unit of real
              evaluation budget. ``overgenerate_factor`` is the inverse of
              the ablation's ``k/N`` (``overgenerate_factor=4`` <->
              ``k/N=0.25``).
            - ``"threshold_reject"``: no over-generation -- scores exactly
              ``n_offspring`` freshly-sampled candidates per generation and
              skips real evaluation for the ones below
              ``threshold_quantile`` of that batch's predicted scores,
              assigning them ``fallback_fitness`` instead. These DO count
              against the evaluation budget (one "slot" each, resolved via
              proxy instead of real eval) -- unlike ``topk_filter``, batch
              size and budget accounting stay unchanged; the saving is pure
              compute avoided per generation.
        feature_mode: Must match the checkpoint's training-time
            ``data_pipeline.py --feature-mode`` exactly. See module
            docstring.
        n_lhs_points: Must match the checkpoint's training-time
            ``data_pipeline.py --lhs-points`` exactly. See module docstring.
        lhs_seed: LHS sampler seed -- also a training/inference match
            requirement (``data_pipeline.py --lhs-seed``).
        overgenerate_factor: ``topk_filter`` only. Pool size per generation
            is ``round(n_offspring * overgenerate_factor)`` (floored to at
            least ``n_offspring``).
        threshold_quantile: ``threshold_reject`` only, in ``[0, 1]``.
            Candidates with a predicted score below this quantile of the
            batch (above, when ``minimization``) are rejected. ``0.5``
            rejects the bottom half of each batch.
        fallback_fitness: Fitness assigned to a ``threshold_reject``-rejected
            or ``topk_filter``-skipped-but-still-logged candidate. ``None``
            uses the running ``LLaMEA.worst_value``.
        include_description / include_configspace: Must match the
            checkpoint's training-time ``data_pipeline.py --no-description``/
            ``--no-configspace`` flags (inverted -- these default to
            included, matching the pipeline's defaults).
        predict_batch_size: Chunk size for ``model.median_point_predictions``
            -- see its docstring for the OOM tradeoffs of raising this.
        num_samples_point_pred: Overrides the checkpoint's
            ``config.num_samples_point_pred`` (median-of-N) if given.
        device: ``"cuda"``/``"cpu"``/``None`` (auto -- whatever
            ``evaluate.load_trained_rlm`` picked).
    """

    checkpoint_dir: str
    strategy: ScreeningStrategy = "topk_filter"
    feature_mode: str = "lhs"
    n_lhs_points: int = 20
    lhs_seed: int = 0
    overgenerate_factor: float = 2.0
    threshold_quantile: float = 0.5
    fallback_fitness: float | None = None
    include_description: bool = True
    include_configspace: bool = True
    predict_batch_size: int = 4
    num_samples_point_pred: int | None = None
    device: str | None = None


class RLMScreener:
    """Loads a trained RLM checkpoint once and scores candidate
    ``Solution`` objects against a fixed set of benchmark problem
    instances.

    The problem-feature text (LHS samples/stats or static meta-properties)
    depends only on the problem instance, never on candidate code, so it is
    computed once per instance at construction time and reused for every
    candidate scored for the lifetime of this object.
    """

    def __init__(
        self, config: RLMScreeningConfig, instances: Sequence[ProblemInstance]
    ):
        from .evaluate import load_trained_rlm

        self.config = config
        self.instances = list(instances)
        if not self.instances:
            raise ValueError(
                "RLMScreener requires at least one ProblemInstance describing "
                "the benchmark functions the LLaMEA run evaluates candidates "
                "against -- pass `rlm_problem_instances` to LLaMEA()."
            )
        self.reg_lm, self.rlm_config = load_trained_rlm(config.checkpoint_dir)
        if config.device:
            self.reg_lm.model.to(config.device)
        self._feature_texts = [
            compute_problem_feature_text(
                inst,
                n_points=config.n_lhs_points,
                seed=config.lhs_seed,
                mode=config.feature_mode,
            )
            for inst in self.instances
        ]

    def score(self, solutions: Sequence[Any]) -> np.ndarray:
        """Returns one predicted score per solution (the mean of its
        per-instance predictions, matching
        ``evaluate.aggregate_instance_predictions``'s training-time
        aggregation)."""
        from .model import median_point_predictions, scalar_point_predictions

        if not solutions:
            return np.array([])

        n_instances = len(self.instances)
        xs: list[str] = []
        for sol in solutions:
            base_x = solution_to_x(
                sol,
                include_description=self.config.include_description,
                include_configspace=self.config.include_configspace,
            )
            xs.extend(f"{base_x}\n\n# Problem\n{ft}" for ft in self._feature_texts)

        num_samples = (
            self.config.num_samples_point_pred or self.rlm_config.num_samples_point_pred
        )
        medians = median_point_predictions(
            self.reg_lm,
            xs,
            num_samples=num_samples,
            config=self.rlm_config,
            batch_size=self.config.predict_batch_size,
        )
        scalars = scalar_point_predictions(medians)
        scalars = np.asarray(scalars, dtype=float).reshape(len(solutions), n_instances)
        return scalars.mean(axis=1)


# --------------------------------------------------------------------------
# Pure selection logic (no torch/regress_lm needed) -- kept separate from
# RLMScreener.score so the filtering policy itself is easy to unit test.
# --------------------------------------------------------------------------


def decide_topk_filter(
    predicted_scores: np.ndarray, n_keep: int, minimization: bool
) -> np.ndarray:
    """Returns a boolean keep-mask selecting the ``n_keep`` best-predicted
    entries (lowest predicted score when ``minimization``, else highest)."""
    n = len(predicted_scores)
    n_keep = max(0, min(n_keep, n))
    order = np.argsort(predicted_scores)
    if not minimization:
        order = order[::-1]
    keep_idx = set(order[:n_keep].tolist())
    return np.array([i in keep_idx for i in range(n)], dtype=bool)


def decide_threshold_reject(
    predicted_scores: np.ndarray, threshold_quantile: float, minimization: bool
) -> np.ndarray:
    """Returns a boolean keep-mask: the predicted scores at or better than
    ``threshold_quantile`` of the batch are kept, the rest rejected."""
    if len(predicted_scores) == 0:
        return np.array([], dtype=bool)
    cutoff = np.quantile(predicted_scores, threshold_quantile)
    if minimization:
        return predicted_scores <= cutoff
    return predicted_scores >= cutoff
