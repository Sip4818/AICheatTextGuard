import optuna
from ._objective import Objective
from ._search_spaces import SearchSpaces


class Tuner:

    def __init__(self, tuning_cfg):
        """
        tuning_cfg: ModelTrainerTuningConfig
        """
        self.cfg = tuning_cfg


        # Shared search space builder
        self.search_spaces = SearchSpaces(cfg=tuning_cfg)


    def tune_xgb_level1(self, X_train, y_train, n_trials=1):
        """
        Tune XGB (Level-1)
        """
        objective = Objective(
            model_name="xgb",
            search_spaces=self.search_spaces,
            cfg=self.cfg.level1.xgb,   # YAML block for Level-1 XGB
            X=X_train,
            y=y_train
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params
    
    def tune_lr_level1(self, X_train, y_train, n_trials=1):
        """
        Tune lr (Level-1)
        """
        objective = Objective(
            model_name="lr",
            search_spaces=self.search_spaces,
            cfg=self.cfg.level1.lr,   # YAML block for Level-1 lr
            X=X_train,
            y=y_train
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    # ---------------------------------------------------------
    # LEVEL-2 LOGISTIC REGRESSION TUNING
    # ---------------------------------------------------------
    def tune_lr_level2(self, X_meta, y_train, n_trials=1):
        """
        Tune Logistic Regression (Level-2 / Meta LR)
        """
        if X_meta is None:
            raise ValueError("Level-2 features must be provided for LR Level-2 tuning.")

        objective = Objective(
            model_name="lr",
            search_spaces=self.search_spaces,
            cfg=self.cfg.level2.lr,   
            X=X_meta,
            y=y_train     
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params
