---
name: hale-vlm-datasets
description: Add, inspect or disable datasets in the Hale-VLM SmolVLM (VLM) and SmolVLA (robotics) registries, and wire them into stage presets and YAML configs. Use when asked to add a new Hugging Face dataset, change a training mixture, or debug a dataset adapter.
---

# Adding a dataset

## VLM dataset

1. In `src/hale_vlm/data/datasets/builtin.py`, add:

```python
@register_dataset("my-dataset")
class MyDatasetAdapter(VLMDataAdapter):          # or VideoDatasetAdapter for video
    spec = DatasetSpec(
        name="my-dataset",
        hf_path="org/my_dataset",
        modality=Modality.IMAGE,                   # IMAGE | MULTI_IMAGE | VIDEO | TEXT
        stage=TrainingStage.VISION,                # VISION | VIDEO | CONTEXT | ...
        category=VisionCategory.CAPTIONING.value,
        description="What it teaches",
        paper_reference="Author et al. (2025)",
        image_fields=("image",),                   # first present field wins
        text_fields=("caption",),
        conversation_field=None,
        config_name=None,                          # HF subset if needed
    )
```

2. Add the name to the right tuple in `data/catalog.py` (`SMOLVLM_VISION_DATASETS`,
   `..._VIDEO_...`, `..._CONTEXT_...`). `SMOLVLM_ALL_DATASETS` is built from them.
3. Override `normalize(row) -> VLMSample | None` only if the generic field lookup can't
   parse the rows. Return `None` to skip a row.
4. Test with `tests/unit/test_dataset_registry.py`-style checks. Assert that it appears in
   `list_datasets(stage=...)` and that `spec.enabled` is set.
5. Update the tables in `README.md` and `docs/data.md`.

To register a dataset but keep it out of training (like `smoltalk`), set
`enabled=False` and put the reason in `description`. `build_dataset` then raises a
clear error.

## Robotics (VLA) dataset

Add a `VLADatasetSpec` in `data/datasets/vla_builtin.py` and the name to
`data/vla_catalog.py` (`SMOLVLA_COMMUNITY_DATASETS`, `..._SIMULATION_...` or
`..._REAL_WORLD_...`). Community paths are listed in `data/vla/community_paths.py`.
Test with `tests/unit/test_vla_registry.py`.

## Quick inspection

```python
from hale_vlm.data import list_datasets
from hale_vlm.data.registry import build_dataset
ad = build_dataset("the-cauldron")
for s in ad.iter_samples(split="train", max_samples=2, streaming=True):
    print(s.modality, s.text[:80], len(s.images))
```
