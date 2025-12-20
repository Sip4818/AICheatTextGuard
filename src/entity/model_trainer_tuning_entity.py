from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LRSpace:
    C: List[float]
    penalty: List[str]
    solver: List[str]
    max_iter: List[int]


@dataclass
class XGBSpace:
    n_estimators: List[int]
    max_depth: List[int]
    learning_rate: List[float]
    subsample: List[float]
    colsample_bytree: List[float]
    min_child_weight: List[int]
    gamma: List[float]
    reg_alpha: List[float]
    reg_lambda: List[float]


@dataclass
class Level1TuningConfig:
    lr: LRSpace
    xgb: XGBSpace


@dataclass
class Level2TuningConfig:
    lr: LRSpace


@dataclass
class ModelTrainerTuningConfig:
    n_trials: int
    level1: Level1TuningConfig
    level2: Level2TuningConfig
