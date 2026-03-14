import optuna
from ._objective import Objective
from ._search_spaces import SearchSpaces
from optuna.integration.mlflow import MLflowCallback
import mlflow
import os

class Tuner:
    def __init__(self, tuning_cfg, tracking_uri="mlruns"):
        """
        tuning_cfg: ModelTrainerTuningConfig
        """
        self.cfg = tuning_cfg
        self.experiment_name = "My_Staking_Model_Project"

        # 1. Ensure the tracking URI is set globally
        mlflow.set_tracking_uri(tracking_uri)

        # 2. Shared search space builder
        self.search_spaces = SearchSpaces(cfg=tuning_cfg)
        
        # 3. Initialize Callback with create_experiment=False
        # This prevents Optuna from fighting your manual mlflow.start_run
        self.mlflc = MLflowCallback(
            tracking_uri=tracking_uri,
            metric_name="AUC Score",
            create_experiment=False,
            mlflow_kwargs={"nested": True}
        )

    def _setup_mlflow(self):
        """
        Ensures the experiment exists and is set as active.
        This fixes the 'Experiment with ID 0 not found' error.
        """
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

    def tune_xgb_level1(self, X_train, y_train, n_trials=1):
        """
        Tune XGB (Level-1)
        """
        self._setup_mlflow()
        
        objective = Objective(
            model_name="xgb",
            search_spaces=self.search_spaces,
            cfg=self.cfg.level1.xgb,
            X=X_train,
            y=y_train,
        )

        with mlflow.start_run(run_name="XGB_Level1_Tuning"):
            study = optuna.create_study(
                study_name="XGB_Level1_Tuning", 
                direction="maximize",
                load_if_exists=True
            )
            study.optimize(objective, n_trials=n_trials, callbacks=[self.mlflc])
            
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_accuracy", study.best_value)

        return study.best_params

    def tune_lr_level1(self, X_train, y_train, n_trials=1):
        """
        Tune lr (Level-1)
        """
        self._setup_mlflow()

        objective = Objective(
            model_name="lr",
            search_spaces=self.search_spaces,
            cfg=self.cfg.level1.lr,
            X=X_train,
            y=y_train,
        )

        with mlflow.start_run(run_name="Lr_Level1_Tuning"):
            study = optuna.create_study(
                study_name="Lr_Level1_Tuning", 
                direction="maximize",
                load_if_exists=True
            )
            study.optimize(objective, n_trials=n_trials, callbacks=[self.mlflc])
            
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_accuracy", study.best_value)

        return study.best_params

    def tune_lr_level2(self, X_meta, y_train, n_trials=1):
        """
        Tune Logistic Regression (Level-2 / Meta LR)
        """
        if X_meta is None:
            raise ValueError("Level-2 features must be provided for LR Level-2 tuning.")

        self._setup_mlflow()

        objective = Objective(
            model_name="lr",
            search_spaces=self.search_spaces,
            cfg=self.cfg.level2.lr,
            X=X_meta,
            y=y_train,
        )

        with mlflow.start_run(run_name="lr_Level2_Tuning"):
            study = optuna.create_study(
                study_name="Lr_Level2_Tuning",
                direction="maximize",
                load_if_exists=True
            )
            study.optimize(objective, n_trials=n_trials, callbacks=[self.mlflc])
            
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_accuracy", study.best_value)

        return study.best_params