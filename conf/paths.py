"""
Centralized path configuration using environment variables.
"""
import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv('GRANT_DATA_DIR', ROOT_DIR / 'data'))
PRETRAINED_DIR = Path(os.getenv('GRANT_PRETRAINED_DIR', ROOT_DIR / 'pretrained'))
OUTPUT_DIR = Path(os.getenv('GRANT_OUTPUT_DIR', ROOT_DIR / 'outputs'))

# Dataset paths
LANGDATA_DIR = DATA_DIR / 'langdata'
SCENEVERSE_DIR = DATA_DIR / 'SceneVerse'

# Model paths
BERT_PATH = os.getenv('GRANT_BERT_PATH', str(PRETRAINED_DIR / 'bert-base-uncased'))
LLM_WEIGHT_PATH = os.getenv('GRANT_LLM_PATH', str(PRETRAINED_DIR / 'llm_weight'))

# Checkpoint paths
POINT_ENCODER_CKPT = os.getenv('GRANT_POINT_ENCODER', '')
GRANT_CKPT = os.getenv('GRANT_CHECKPOINT', '')

# Cluster-specific (optional)
LD_LIBRARY_PATH = os.getenv('LD_LIBRARY_PATH', '')

def get_scanrefer_path(dataset_name='ScanRefer_filtered_full_withroot_addeval'):
    """Get ScanRefer dataset path with optional dataset name."""
    return LANGDATA_DIR / 'scanrefer' / f'{dataset_name}.json'

def validate_paths():
    """Validate that all required paths exist."""
    required_paths = {
        'Data directory': DATA_DIR,
        'Pretrained directory': PRETRAINED_DIR,
        'BERT model': BERT_PATH,
    }

    missing = []
    for name, path in required_paths.items():
        if not Path(path).exists():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing required paths:\n" + "\n".join(missing)
        )

    return True