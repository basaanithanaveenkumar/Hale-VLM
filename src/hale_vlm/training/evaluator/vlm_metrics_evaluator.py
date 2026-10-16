"""VLMMetricsEvaluator: Evaluator that aggregates any set of VLMMetricBase metrics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from hale_vlm.metrics import (
    LossMetric,
    VLMMetricBase,
)


class VLMMetricsEvaluator:
    """Evaluator that aggregates any set of VLMMetricBase metrics.

    Usage example::

        evaluator = VLMMetricsEvaluator(metrics=[
            LossMetric(),
            BLEUMetric(),
            ROUGEMetric(),
            ExactMatchMetric(),
        ])

        # generation_fn: model + batch → dict(loss, generated_texts, references)
        results = evaluator.run(model, val_loader, generation_fn=my_gen_fn)
        # results = {"loss": 2.1, "bleu_4": 0.28, "rouge_l_f": 0.51, ...}

    The ``generation_fn`` argument is optional.  When provided it must have
    the signature::

        def generation_fn(model, batch) -> dict:
            # Must return at least "loss" (float).
            # For text metrics: "generated" (str) and "reference" (str).
            # For VQA metrics: "prediction" (str) and "answers" (list[str]).
            # Keys beyond "loss" are passed to each metric's update(**kwargs).

    Metrics whose required kwargs are not present in the generation_fn output
    are silently skipped for that batch (update is not called).
    """

    def __init__(self, metrics: Sequence[VLMMetricBase] | None = None) -> None:
        self.metrics: list[VLMMetricBase] = list(metrics or [LossMetric()])

    def run(self, model, loader, generation_fn=None) -> dict[str, float]:
        """Run evaluation and return aggregated metrics.

        Args:
            model:         The VLM model to evaluate.
            loader:        DataLoader yielding batches.
            generation_fn: Optional callable(model, batch) → dict of outputs.
                           When None, only the forward loss is measured.
        """
        model.eval()
        for m in self.metrics:
            m.reset()

        with torch.no_grad():
            for batch in loader:
                if generation_fn is not None:
                    outputs = generation_fn(model, batch)
                else:
                    fwd = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    outputs = {"loss": float(fwd.loss.item())}

                # Each metric inspects its own update() signature and picks
                # only the kwargs it needs from the output dict.
                for m in self.metrics:
                    import inspect

                    sig = inspect.signature(m.update)
                    # Collect only kwargs the metric's update() accepts
                    valid_kwargs = {k: v for k, v in outputs.items() if k in sig.parameters}
                    if valid_kwargs:
                        try:
                            m.update(**valid_kwargs)
                        except TypeError:
                            pass  # missing required kwarg — skip this metric for batch

        model.train()

        # Merge all metric results into a single flat dict
        results: dict[str, float] = {}
        for m in self.metrics:
            results.update(m.compute())
        return results
