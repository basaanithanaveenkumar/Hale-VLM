"""Hale-VLM exception hierarchy.

.. code-block:: text

    HaleCoreError
    └── HaleVLMError
        ├── VLMConfigError            (also ConfigError, ValueError)
        ├── VLMInputError             (also ValueError)
        ├── VLMDatasetError           (also DataError, ValueError)
        │   └── DatasetUnavailableError
        └── UnsupportedOperationError (also NotImplementedError)
"""

from hale_vlm.exceptions.dataset_unavailable_error import DatasetUnavailableError
from hale_vlm.exceptions.hale_vlm_error import HaleVLMError
from hale_vlm.exceptions.unsupported_operation_error import UnsupportedOperationError
from hale_vlm.exceptions.vlm_config_error import VLMConfigError
from hale_vlm.exceptions.vlm_dataset_error import VLMDatasetError
from hale_vlm.exceptions.vlm_input_error import VLMInputError

__all__ = [
    "DatasetUnavailableError",
    "HaleVLMError",
    "UnsupportedOperationError",
    "VLMConfigError",
    "VLMDatasetError",
    "VLMInputError",
]
