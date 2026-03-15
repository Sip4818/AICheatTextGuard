import os
import numpy as np
from typing import Any
from dataclasses import asdict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils.logger import logger
from src.utils.exception import AITextException
from src.utils.common import (
    save_object,
    read_object,
    save_numpy,
    log_file_size,
    assert_file_exists,
    read_numpy,
    write_yaml,
    extract_params,
)
from src.entity.config_entity import ModelTrainerConfig
from src.entity.model_trainer_tuning_entity import ModelTrainerTuningConfig
from src.entity.model_trainer_final_params_entity import ModelTrainerFinalParamsConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.tuning.tuner import Tuner
from src.model.stack_model import StackedModel
from src.constants.constants import (
    SEED,
    LR_KEYS,
    XGB_KEYS,
    params_dict_format,
    params_yaml_file_path,
)

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        model_trainer_tuning_config: ModelTrainerTuningConfig,
        model_trainer_final_params_config: ModelTrainerFinalParamsConfig,
    ) -> None:
        self.model_trainer_config = model_trainer_config
        self.model_trainer_tuning_config = model_trainer_tuning_config
        self.model_trainer_final_params_config = model_trainer_final_params_config
        preprocessor = read_object(self.model_trainer_config.preprocessing_object_path)
        self.object_storage = {"preprocessor": preprocessor}

    def _log_metrics(self, fold_idx: Any, y_true: np.ndarray, y_prob: np.ndarray, prefix: str = "") -> float:
        """Helper to calculate and log multiple classification metrics."""
        y_pred = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y_true, y_prob)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        logger.info(
            f"{prefix} Fold {fold_idx} -> AUC: {auc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1: {f1:.4f}"
        )
        return auc

    def _generate_oof_xgb_preds(self, X: np.ndarray, y: np.ndarray, best_params_xgb_lvl1: dict) -> np.ndarray:
        try:
            N_FOLDS = self.model_trainer_config.folds
            oof_preds_xgb = np.zeros(X.shape[0])
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            
            for i, (train_index, val_index) in enumerate(skf.split(X, y)):
                logger.info(f"--- XGB Fold {i + 1} started ----")

                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                xgb = XGBClassifier(**best_params_xgb_lvl1)
                xgb.fit(X_train, y_train)

                y_pred_proba = xgb.predict_proba(X_val)[:, 1]
                oof_preds_xgb[val_index] = y_pred_proba

                self._log_metrics(i + 1, y_val, y_pred_proba, prefix="XGB")

            save_numpy(oof_preds_xgb, self.model_trainer_config.xgb_level1_oof_predictions_path)
            assert_file_exists(self.model_trainer_config.xgb_level1_oof_predictions_path, "OOF xgb numpy")
            log_file_size(self.model_trainer_config.xgb_level1_oof_predictions_path, "OOF xgb numpy")

            logger.info("\n" + "="*25 + " OVERALL XGB OOF METRICS " + "="*25)
            self._log_metrics("OVERALL", y, oof_preds_xgb, prefix="XGB")
            logger.info("="*75 + "\n")
            
            return oof_preds_xgb
        except Exception as e:
            logger.error("Could not generate oof xgb predictions")
            raise AITextException(e)

    def _generate_oof_lr_preds(self, X: np.ndarray, y: np.ndarray, best_params_lr_lvl1: dict) -> np.ndarray:
        try:
            N_FOLDS = self.model_trainer_config.folds
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            oof_preds_lr = np.zeros(X.shape[0])

            for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"--- LR Fold {i + 1} started ---")

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                pipe = Pipeline([
                    ("Scaler", StandardScaler()),
                    ("lr", LogisticRegression(**best_params_lr_lvl1)),
                ])

                pipe.fit(X_train, y_train)

                y_pred_proba = pipe.predict_proba(X_val)[:, 1]
                oof_preds_lr[val_idx] = y_pred_proba

                self._log_metrics(i + 1, y_val, y_pred_proba, prefix="LR")

            save_numpy(oof_preds_lr, self.model_trainer_config.lr_level1_oof_predictions_path)
            assert_file_exists(self.model_trainer_config.lr_level1_oof_predictions_path, "OOF lr numpy")
            log_file_size(self.model_trainer_config.lr_level1_oof_predictions_path, "OOF lr numpy")

            logger.info("\n" + "="*25 + " OVERALL LR OOF METRICS " + "="*25)
            self._log_metrics("OVERALL", y, oof_preds_lr, prefix="LR")
            logger.info("="*75 + "\n")
            
            return oof_preds_lr
        except Exception as e:
            logger.error("Could not generate oof lr predictions")
            raise AITextException(e)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oof_preds_xgb: np.ndarray,
        oof_preds_lr: np.ndarray,
        best_params_xgb_lvl1: dict,
        best_params_lr_lvl1: dict,
        best_params_lr_lvl2: dict,
    ):
        try:
            logger.info("Training the final Level 1 and Meta models on full dataset")
            
            # Final XGB Level 1
            xgb_lvl1 = XGBClassifier(**best_params_xgb_lvl1)
            xgb_lvl1.fit(X, y)
            save_object(xgb_lvl1, self.model_trainer_config.xgb_level1_model_path)
            self.object_storage["xgb_model"] = xgb_lvl1

            # Final LR Level 1
            lr_lvl1_pipeline = Pipeline([
                ("Scaler", StandardScaler()),
                ("lr", LogisticRegression(**best_params_lr_lvl1)),
            ])
            lr_lvl1_pipeline.fit(X, y)
            save_object(lr_lvl1_pipeline, self.model_trainer_config.lr_level1_model_path)
            self.object_storage["lr_model"] = lr_lvl1_pipeline

            # Meta Model Training (Level 2)
            X_meta = np.column_stack([oof_preds_xgb, oof_preds_lr])
            lr_lvl2_pipeline = Pipeline([
                ("Scaler", StandardScaler()),
                ("lr", LogisticRegression(**best_params_lr_lvl2)),
            ])
            lr_lvl2_pipeline.fit(X_meta, y)
            save_object(lr_lvl2_pipeline, self.model_trainer_config.meta_lr_path)
            self.object_storage["meta_model"] = lr_lvl2_pipeline
            
        except Exception as e:
            logger.error("Error occurred while training final models")
            raise AITextException(e)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logger.info("Initiating model trainer")

            data = read_numpy(self.model_trainer_config.transformed_train_data_path)
            X, y = data[:, :-1], data[:, -1]

            if self.model_trainer_config.enable_tuning:
                n_trials = self.model_trainer_tuning_config.n_trials
                tuner = Tuner(tuning_cfg=self.model_trainer_tuning_config)

                best_params_lr_lvl1 = tuner.tune_lr_level1(X, y, n_trials)
                lvl1_lr_params_cleaned = extract_params(best_params_lr_lvl1, LR_KEYS)
                
                best_params_xgb_lvl1 = tuner.tune_xgb_level1(X, y, n_trials)
                lvl1_xgb_params_cleaned = extract_params(best_params_xgb_lvl1, XGB_KEYS)

                params_dict_format["model_trainer"]["level1"]["lr"] = lvl1_lr_params_cleaned
                params_dict_format["model_trainer"]["level1"]["xgb"] = lvl1_xgb_params_cleaned

                oof_preds_xgb = self._generate_oof_xgb_preds(X, y, best_params_xgb_lvl1)
                oof_preds_lr = self._generate_oof_lr_preds(X, y, best_params_lr_lvl1)

                X_meta = np.column_stack([oof_preds_xgb, oof_preds_lr])
                best_params_lr_lvl2 = tuner.tune_lr_level2(X_meta, y, n_trials)
                lvl2_lr_params_cleaned = extract_params(best_params_lr_lvl2, LR_KEYS)
                
                params_dict_format["model_trainer"]["level2"]["lr"] = lvl2_lr_params_cleaned
                write_yaml(params_dict_format, params_yaml_file_path)

            else:
                best_params_lr_lvl1 = asdict(self.model_trainer_final_params_config.level1.lr)
                best_params_xgb_lvl1 = asdict(self.model_trainer_final_params_config.level1.xgb)
                best_params_lr_lvl2 = asdict(self.model_trainer_final_params_config.level2.lr)

                oof_preds_xgb = self._generate_oof_xgb_preds(X, y, best_params_xgb_lvl1)
                oof_preds_lr = self._generate_oof_lr_preds(X, y, best_params_lr_lvl1)

            self.train(
                X, y, oof_preds_xgb, oof_preds_lr,
                best_params_xgb_lvl1, best_params_lr_lvl1, best_params_lr_lvl2,
            )
            
            final_stacked_model = StackedModel(**self.object_storage)
            save_object(final_stacked_model, self.model_trainer_config.final_model_path)

            logger.info(f"Model trainer stage completed. Final model saved at: {self.model_trainer_config.final_model_path}")

            return ModelTrainerArtifact(
                model_path=os.path.dirname(self.model_trainer_config.final_model_path)
            )
        except Exception as e:
            logger.error("Model Trainer stopped due to an error")
            raise AITextException(e)