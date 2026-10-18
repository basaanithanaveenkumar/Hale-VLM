"""The Hale-VLM exception hierarchy roots at HaleCoreError and stays catchable as built-ins."""

import pytest
from hale_core.exceptions import ConfigError, DataError, HaleCoreError

from hale_vlm.exceptions import (
    DatasetUnavailableError,
    HaleVLMError,
    UnsupportedOperationError,
    VLMConfigError,
    VLMDatasetError,
    VLMInputError,
)


@pytest.mark.parametrize(
    ("exc", "builtin"),
    [
        (VLMConfigError, ValueError),
        (VLMInputError, ValueError),
        (VLMDatasetError, ValueError),
        (DatasetUnavailableError, ValueError),
        (UnsupportedOperationError, NotImplementedError),
    ],
)
def test_hierarchy(exc, builtin):
    assert issubclass(exc, HaleVLMError)
    assert issubclass(exc, HaleCoreError)
    assert issubclass(exc, builtin)


def test_specific_core_bases():
    assert issubclass(VLMConfigError, ConfigError)
    assert issubclass(VLMDatasetError, DataError)
    assert issubclass(DatasetUnavailableError, VLMDatasetError)


def test_invalid_connector_dims_raise_vlm_input_error():
    from hale_vlm.vision.connectors import QFormer

    with pytest.raises(VLMInputError, match="divisible"):
        QFormer(8, 16, hidden_dim=10, num_heads=4)
