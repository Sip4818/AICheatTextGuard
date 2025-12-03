from datetime import datetime
import os
from src.constants.constants import data_root, artifacts_root
class TrainingPipelineConfig:
    def __init__(self):
        self.data_root: str= data_root
        self.artifact_dir: str= artifacts_root
        