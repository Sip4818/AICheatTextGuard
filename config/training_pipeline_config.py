from datetime import datetime
import os
from src.constants.constants import data_root, artifacts_root
class TrainingPipelineConfig:
    def __init__(self):
        self.data_root: str= data_root
        self.artifact_root: str= artifacts_root
        self.timestamp: str=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir=os.path.join(self.artifact_root,self.timestamp)
        