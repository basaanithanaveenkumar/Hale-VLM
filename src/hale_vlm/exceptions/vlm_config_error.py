"""Invalid Hale-VLM configuration."""

from __future__ import annotations

from hale_core.exceptions import ConfigError

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class VLMConfigError(HaleVLMError, ConfigError):
    """A VLM config value is inconsistent or unsupported (unknown connector, bad dtype, empty list).

    Also a :class:`ValueError` (through :class:`ConfigError`) so pre-existing handlers keep working.
    """
