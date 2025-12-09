from csv import field_size_limit
import logging

from git import exc
from src.utils.logger import logger
from src.utils.exception import AITextException
from src.utils.common import save_model, save_numpy, log_file_size, assert_file_exists, read_numpy
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.constants.constants  import SEED

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self, model_trainer_config :ModelTrainerConfig)->ModelTrainerArtifact:
        self.model_trainer_config=model_trainer_config
  
    def _generate_oof_xgb_preds(self,X,y,best_params_xgb_lvl1) -> np.ndarray:
        try:
            N_FOLDS = self.model_final_params_config.folds
            oof_preds_xgb = np.zeros(X.shape[0])
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for i, (train_index, val_index) in enumerate(skf.split(X, y)):
                logger.info(f"--- Fold {i} started ---")

                X_train,X_val=X[train_index],X[val_index]
                y_train,y_val=y[train_index],y[val_index]

                xgb=XGBClassifier(**best_params_xgb_lvl1)
                xgb.fit(X_train,y_train)

                y_pred_proba = xgb.predict_proba(X_val)[:, 1]

                oof_preds_xgb[val_index] = y_pred_proba 

                auc_score = roc_auc_score(y_val, y_pred_proba)

                print(f"Fold {i} AUC score is {auc_score:.4f}")
            
            save_numpy(oof_preds_xgb,self.model_trainer_config.xgb_level1_model_path )
            assert_file_exists(self.model_trainer_config.xgb_level1_model_path, "OOF lr numpy")
            log_file_size(self.model_trainer_config.xgb_level1_model_path, "OOF lr numpy")

            overall_oof_auc = roc_auc_score(y, oof_preds_xgb)
            logger.info(f"\nOverall OOF AUC: {overall_oof_auc:.4f}")
            return oof_preds_xgb
        except Exception as e:
            logging.error("Could not generate oof xgb predictions")
            AITextException(e)
    
    def _generate_oof_lr_preds(self,X,y,best_params_lr_lvl1) -> np.ndarray :
        try:
            N_FOLDS = self.model_final_params_config.folds

            skf=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            oof_preds_lr = np.zeros(X.shape[0]) # Initialize OOF array with size of X

            for i,(train_idx,val_idx) in enumerate(skf.split(X, y)):
                print(f"--- Fold {i} started ---")

                X_train,X_val=X[train_idx],X[val_idx]
                y_train,y_val=y[train_idx],y[val_idx]

                ss=StandardScaler()
                X_train=ss.fit_transform(X_train)
                X_val=ss.transform(X_val)
                
                lr=LogisticRegression(**best_params_lr_lvl1)
                lr.fit(X_train,y_train)

                y_pred_proba = lr.predict_proba(X_val)[:, 1]
                oof_preds_lr[val_idx] = y_pred_proba 

                auc_score = roc_auc_score(y_val, y_pred_proba)
                logger.info(f"Fold {i} AUC score is {auc_score:.4f}")

            save_numpy(oof_preds_lr,self.model_trainer_config.lr_level1_model_path )
            assert_file_exists(self.model_trainer_config.lr_level1_model_path, "OOF xgb numpy")
            log_file_size(self.model_trainer_config.lr_level1_model_path, "OOF xgb numpy")            overall_oof_auc = roc_auc_score(y, oof_preds_lr)
            logger.info(f"\nOverall OOF AUC: {overall_oof_auc:.4f}")
            return oof_preds_lr
        except Exception as e:
            logger.error("Could not generate oof lr predictions")
            raise AITextException(e)
    
    def train(self,X,y,oof_preds_xgb,oof_preds_lr,best_params_xgb_lvl1, best_params_lr_lvl1,best_params_lr_lvl2):
        try:
            logger.info("Training the models")
            xgb_lvl1=XGBClassifier(**best_params_xgb_lvl1)
            xgb_lvl1.fit(X,y)
            save_model(xgb_lvl1,self.model_trainer_config.xgb_level1_model_path)
            
            lr_lvl1=LogisticRegression(**best_params_lr_lvl1)
            lr_lvl1.fit(X,y)
            save_model(lr_lvl1, self.model_trainer_config.lr_level1_model_path)

            lr_lvl2=LogisticRegression(**best_params_lr_lvl2)
            lvl2_train = np.vstack([oof_preds_xgb, oof_preds_lr]).T
            lr_lvl2.fit(lvl2_train)
            save_model(lr_lvl2, self.model_trainer_config.meta_lr_path)
        except Exception as e:
            logger.error("Error occured while training")
            raise AITextException(e)

    def initiate_model_training(self):
        try:
            logger.info("Initiating model trainer")

            data = read_numpy(self.model_trainer_config.transformed_train_data_path)

            X,y =data[:,:-1], data[:,-1]
            
            best_params_lr_lvl1=self.lr_lvl1_hp_tuning(X,y)
            best_params_xgb_lvl1=self.xgb_lvl1_hp_tuning(X,y)
            oof_preds_xgb=self._generate_oof_xgb_preds(X,y,best_params_xgb_lvl1)
            oof_preds_lr=self._generate_oof_lr_preds(X,y,best_params_lr_lvl1)

            best_params_lr_lvl2=self.lr_lvl2_hp_tuning(oof_preds_xgb,oof_preds_lr,y)
            self.train(X,y,oof_preds_xgb,oof_preds_lr,best_params_xgb_lvl1,best_params_lr_lvl1,best_params_lr_lvl2)
        except Exception as e:
            logger.error("Model Trainer stopped due to some error")
            raise AITextException(e)
