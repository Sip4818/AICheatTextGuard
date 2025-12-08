from src.utils.logger import logger
from src.utils.exception import AITextException
from src.utils.common import save_model
from src.entity.config_entity import ModelTrainerConfig
from src.entity.model_trainer_tuning_entity import ModelTrainerTuningConfig
from src.entity.model_trainer_final_params_entity import ModelTrainerFinalParamsConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from src.constants.constants  import SEED
import numpy as np
import pandas as pd
import optuna
class ModelTrainer:
    def __init__(self, model_trainer_config :ModelTrainerConfig,
                  model_tuning_config: ModelTrainerTuningConfig,
                   model_final_params_config: ModelTrainerFinalParamsConfig)->ModelTrainerArtifact:
        
        self.model_trainer_config=model_trainer_config
        self.mode_tuning_config=model_tuning_config
        self.model_final_params_config=model_final_params_config

    def lr_lvl1_hp_tuning(self):
        pass
    def xgb_lvl1_hp_tuning(self):
        pass
    def lr_lvl2_hp_tuning(self):
        pass


    def generate_oof_xgb_preds(self,X,y,best_params_xgb_lvl1):
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
            np.save(self.model_trainer_config.xgb_level1_model_path, oof_preds_xgb)

            overall_oof_auc = roc_auc_score(y, oof_preds_xgb)
            logger.info(f"\nOverall OOF AUC: {overall_oof_auc:.4f}")
            return oof_preds_xgb
        except Exception as e:
            AITextException(e)
    
    def generate_oof_lr_preds(self,X,y,best_params_lr_lvl1):
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

            np.save(self.model_trainer_config.lr_level1_model_path, oof_preds_lr)
            overall_oof_auc = roc_auc_score(y, oof_preds_lr)
            logger.info(f"\nOverall OOF AUC: {overall_oof_auc:.4f}")
            return oof_preds_lr
        except Exception as e:
            AITextException(e)
    
    def train(self,X,y,oof_preds_xgb,oof_preds_lr,best_params_xgb_lvl1, best_params_lr_lvl1,best_params_lr_lvl2):
        
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

    def initiate_model_training(self):
        data = np.load('train.npy') 
        X,y =data[:,:-1], data[:,-1]
        best_params_lr_lvl1=self.lr_lvl1_hp_tuning(X,y)
        best_params_xgb_lvl1=self.xgb_lvl1_hp_tuning(X,y)
        oof_preds_xgb=self.generate_oof_xgb_preds(X,y,best_params_xgb_lvl1)
        oof_preds_lr=self.generate_oof_lr_preds(X,y,best_params_lr_lvl1)

        best_params_lr_lvl2=self.lr_lvl2_hp_tuning(oof_preds_xgb,oof_preds_lr,y)
        self.train(X,y,oof_preds_xgb,oof_preds_lr,best_params_xgb_lvl1,best_params_lr_lvl1,best_params_lr_lvl2)
