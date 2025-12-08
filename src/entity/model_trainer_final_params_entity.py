


from dataclasses import dataclass

@dataclass
class LRFinalParams:
    C: float
    penalty: str
    solver: str
    max_iter: int


@dataclass
class XGBFinalParams:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    gamma: float
    reg_alpha: float
    reg_lambda: float


@dataclass
class Level1FinalParams:
    lr: LRFinalParams
    xgb: XGBFinalParams


@dataclass
class Level2FinalParams:
    lr: LRFinalParams


@dataclass
class ModelTrainerFinalParamsConfig:
    folds: int
    level1: Level1FinalParams
    level2: Level2FinalParams
