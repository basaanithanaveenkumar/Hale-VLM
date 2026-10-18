import pytest


@pytest.mark.smoke
def test_imports():
    import hale_vlm  # noqa: F401
    from hale_vlm.config import VLMRunConfig, load_vlm_config
    from hale_vlm.models.vlm import VLM_VARIANTS, HaleVLM
    from hale_vlm.vision.projector import VisionProjector

    assert hale_vlm.__version__ == "0.1.0"
    assert "qwen3_8b_vlm" in VLM_VARIANTS
    assert VLMRunConfig is not None
    assert HaleVLM is not None
    assert VisionProjector is not None
    assert load_vlm_config is not None


@pytest.mark.smoke
def test_package_imports_without_optional_extras():
    """Importing the package must not pull ray, webdataset, onnx, matplotlib or MLX."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import hale_vlm, hale_vlm.config, hale_vlm.models, hale_vlm.data, hale_vlm.vision\n"
        "import hale_vlm.cli.train, hale_vlm.cli.chat, hale_vlm.deploy\n"
        "heavy = {'ray', 'webdataset', 'onnx', 'onnxruntime', 'matplotlib', 'mlx', 'tensorrt'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-800:]
