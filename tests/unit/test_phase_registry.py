"""Phase registries resolve by name, filter by phase and are wired into the data config."""

import pytest

from hale_vlm.config.sections.data import VLMDataConfig
from hale_vlm.data.phase_registry import list_phase_datasets
from hale_vlm.data.registry import get_dataset
from hale_vlm.data.types import TrainingPhase
from hale_vlm.data.vla.phase_registry import list_vla_phase_datasets
from hale_vlm.data.vla_registry import build_vla_dataset, get_vla_dataset
from hale_vlm.exceptions import DatasetUnavailableError


@pytest.mark.parametrize("phase", list(TrainingPhase))
def test_every_phase_has_datasets(phase):
    assert list_phase_datasets(phase=phase)


def test_phase_dataset_resolves_through_builtin_lookup():
    assert get_dataset("cc3m").spec.name == "cc3m"


def test_vla_phase_datasets_register_and_filter():
    pretrain = list_vla_phase_datasets(phase="pretrain")
    assert pretrain
    assert len(list_vla_phase_datasets()) >= len(pretrain)
    assert get_vla_dataset("agibot-world").spec.name == "agibot-world"


def test_locally_collected_vla_dataset_cannot_be_streamed():
    adapter = build_vla_dataset("apo-interventions")
    with pytest.raises(DatasetUnavailableError, match="collected locally"):
        next(iter(adapter.open_stream()))


def test_phase_registry_source_resolves_datasets():
    cfg = VLMDataConfig(source="phase_registry", registry_phase="mid_train")
    assert cfg.resolved_registry_datasets() == list_phase_datasets(phase=TrainingPhase.MID_TRAIN)
    assert cfg.uses_registry_stream()


def test_phase_registry_source_requires_phase():
    with pytest.raises(ValueError, match="registry_phase"):
        VLMDataConfig(source="phase_registry")
