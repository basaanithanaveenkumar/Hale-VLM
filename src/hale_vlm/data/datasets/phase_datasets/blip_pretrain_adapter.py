"""Registry adapter for the ``blip-laion-cc-sbu-558k`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("blip-laion-cc-sbu-558k")
class BLIPPretrainAdapter(VLMDataAdapter):
    """Registry adapter for ``blip-laion-cc-sbu-558k``.

    Source: Hub dataset ``BleachNick/BLIP_laion_cc_sbu_558k``.

    BLIP pretraining LAION+CC3M+SBU mixture.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="blip-laion-cc-sbu-558k",
        hf_path="BleachNick/BLIP_laion_cc_sbu_558k",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="BLIP pretraining LAION+CC3M+SBU mixture.",
        paper_reference="Li et al. (2022) BLIP",
        image_fields=("image",),
        text_fields=("caption", "text"),
        conversation_field=None,
    )
