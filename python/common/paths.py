from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_01_DIR = DATA_DIR / "sample_01"
ICBHI_2017_DIR = SAMPLE_01_DIR / "ICBHI_final_database"
ICBHI_2017_LABELS = SAMPLE_01_DIR / "labels.txt"
SAMPLES_DIR = DATA_DIR / "samples"
ICBHI_DIR = SAMPLES_DIR / "ICBHI_final_database"
ICBHI_LABELS = SAMPLES_DIR / "labels.txt"
SAMPLES_02_DIR = DATA_DIR / "samples_02"
COMBINED_DIR = DATA_DIR / "combined"
COMBINED_AUDIO_DIR = COMBINED_DIR / "audio"
COMBINED_LABELS = COMBINED_DIR / "labels.csv"
PROCESSED_AUDIO_DIR = COMBINED_DIR / "processed_audio"
SPECTROGRAMS_DIR = COMBINED_DIR / "spectrograms"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAINING_ARTIFACTS_DIR = ARTIFACTS_DIR / "training"
QUANTIZATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "quantization"
COMPILE_ARTIFACTS_DIR = ARTIFACTS_DIR / "compile"
PAPER_FIGURES_DIR = ARTIFACTS_DIR / "paper_figures"
FEATURE_ARTIFACTS_DIR = ARTIFACTS_DIR / "features"

KD_PIPELINE_ARTIFACTS_DIR = TRAINING_ARTIFACTS_DIR / "kd_cnn_bilstm_to_cnn"
ICBHI_3CLASS_KD_ARTIFACTS_DIR = TRAINING_ARTIFACTS_DIR / "icbhi_3class_kd"
LAYER13_ARTIFACTS_DIR = TRAINING_ARTIFACTS_DIR / "layer1_3"
CALIBRATION_DATA_DIR = QUANTIZATION_ARTIFACTS_DIR / "calibration_data"
VITIS_QAT_ARTIFACTS_DIR = QUANTIZATION_ARTIFACTS_DIR / "vitis_qat_v3"
VITIS_AI_FLOW_ARTIFACTS_DIR = QUANTIZATION_ARTIFACTS_DIR / "vitis_ai_flow"


def repo_path(*parts: PathLike) -> Path:
    return PROJECT_ROOT.joinpath(*map(str, parts))


def artifact_path(*parts: PathLike) -> Path:
    return ARTIFACTS_DIR.joinpath(*map(str, parts))


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_common_path(current_file: str, levels_to_python_dir: int = 1) -> None:
    """Deprecated compatibility shim; import paths.py directly when possible."""
    return None
