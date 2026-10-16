"""VLA phase-tagged dataset registrations for Hale-VLM (all three phases).

All adapters here are registered in VLA_PHASE_DATASETS (see vla/phase_registry.py),
NOT in the SmolVLA VLA_DATASETS registry, so existing tests are unaffected.

Phase 1 — PRETRAIN (18 datasets):
  Core cross-embodiment:
    open-x-embodiment, bridge-v2, fractal-rt1, bc-z,
    droid-v1, libero-pretrain
  Aggregated packs:
    openEAI-dataset, lerobot-community-v3, robogene
  Human-video / tactile:
    being-h0, agibot-world, h-tac-ttp
  Synthetic / simulation:
    syngrasp-1b, robocasa
  Egocentric human video (cross-embodiment bridge):
    ego4d, vitra, egovla

Phase 2 — MID_TRAIN (5 datasets, embodied VLM — no action labels):
  refspatial, embspatial-bench, robo2vlm, robopoint, vln-r2r

Phase 3 — POST_TRAIN (4 local-collection datasets):
  flowpro-pairs, apo-interventions, hindsight-relabeled, dagger-corrections
"""

from __future__ import annotations

from .agibot_world_adapter import AgibotWorldAdapter
from .apo_interventions_adapter import APOInterventionsAdapter
from .bcz_adapter import BCZAdapter
from .being_h0_adapter import BeingH0Adapter
from .bridge_v2_adapter import BridgeV2Adapter
from .d_agger_corrections_adapter import DAggerCorrectionsAdapter
from .droid_v1_adapter import DroidV1Adapter
from .ego4_d_adapter import Ego4DAdapter
from .ego_vla_adapter import EgoVLAAdapter
from .emb_spatial_bench_adapter import EmbSpatialBenchAdapter
from .flow_pro_pairs_adapter import FlowPROPairsAdapter
from .fractal_rt1_adapter import FractalRT1Adapter
from .h_tac_ttp_adapter import HTacTTPAdapter
from .hindsight_relabeled_adapter import HindsightRelabeledAdapter
from .le_robot_community_v3_adapter import LeRobotCommunityV3Adapter
from .libero_pretrain_adapter import LiberoPretrainAdapter
from .open_eai_dataset_adapter import OpenEAIDatasetAdapter
from .open_x_embodiment_adapter import OpenXEmbodimentAdapter
from .ref_spatial_adapter import RefSpatialAdapter
from .robo2_vlm_adapter import Robo2VLMAdapter
from .robo_casa_adapter import RoboCasaAdapter
from .robo_gene_adapter import RoboGeneAdapter
from .robo_point_adapter import RoboPointAdapter
from .syn_grasp1_b_adapter import SynGrasp1BAdapter
from .vitra_adapter import VITRAAdapter
from .vlnr2_r_adapter import VLNR2RAdapter

__all__ = [
    "APOInterventionsAdapter",
    "AgibotWorldAdapter",
    "BCZAdapter",
    "BeingH0Adapter",
    "BridgeV2Adapter",
    "DAggerCorrectionsAdapter",
    "DroidV1Adapter",
    "Ego4DAdapter",
    "EgoVLAAdapter",
    "EmbSpatialBenchAdapter",
    "FlowPROPairsAdapter",
    "FractalRT1Adapter",
    "HTacTTPAdapter",
    "HindsightRelabeledAdapter",
    "LeRobotCommunityV3Adapter",
    "LiberoPretrainAdapter",
    "OpenEAIDatasetAdapter",
    "OpenXEmbodimentAdapter",
    "RefSpatialAdapter",
    "Robo2VLMAdapter",
    "RoboCasaAdapter",
    "RoboGeneAdapter",
    "RoboPointAdapter",
    "SynGrasp1BAdapter",
    "VITRAAdapter",
    "VLNR2RAdapter",
]
