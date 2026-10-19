"""Exception hierarchy of :mod:`hale_vlm`.

.. code-block:: text

    hale_core.exceptions.HaleCoreError
    └── HaleVLMError
        ├── DatasetError               (also ValueError)
        ├── ModelConfigError           (also ValueError)
        ├── UnsupportedOperationError  (also NotImplementedError)
        └── FilmError                  (also ValueError)

Config files and registries raise the shared core errors (``hale_core.exceptions.ConfigError``,
``RegistryError``) and deployment raises ``hale_core.deploy`` errors; pydantic validators keep
raising ``ValueError`` because pydantic turns those into ``ValidationError``.
"""

from hale_vlm.exceptions.dataset_error import DatasetError
from hale_vlm.exceptions.film_error import FilmError
from hale_vlm.exceptions.hale_vlm_error import HaleVLMError
from hale_vlm.exceptions.model_config_error import ModelConfigError
from hale_vlm.exceptions.unsupported_operation_error import UnsupportedOperationError

__all__ = [
    "DatasetError",
    "FilmError",
    "HaleVLMError",
    "ModelConfigError",
    "UnsupportedOperationError",
]
