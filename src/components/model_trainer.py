from src.utils.logger import logger
from src.utils.exception import AITextException
from src.utils.common import save_object,read_object, save_numpy, log_file_size, assert_file_exists, read_numpy
from src.entity.config_entity import ModelTrainerConfig
from src.entity.model_trainer_tuning_entity import ModelTrainerTuningConfig
from src.entity.model_trainer_final_params_entity import ModelTrainerFinalParamsConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.tuning.tuner import Tuner
from src.model.stack_model import StackedModel
from src.constants.constants  import SEED

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self, model_trainer_config :ModelTrainerConfig,
                  model_trainer_tuning_config: ModelTrainerTuningConfig, 
                  model_trainer_final_params_config: ModelTrainerFinalParamsConfig)->None:
        
        self.model_trainer_config=model_trainer_config
        self.model_trainer_tuning_config= model_trainer_tuning_config
        self.model_trainer_final_params_config = model_trainer_final_params_config
        preprocessor= read_object(self.model_trainer_config.preprocessing_object_path)
        self.object_storage={"preprocessor": preprocessor}
  
    def _generate_oof_xgb_preds(self,X,y,best_params_xgb_lvl1) -> np.ndarray:
        try:
            N_FOLDS = self.model_trainer_final_params_config.folds
            oof_preds_xgb = np.zeros(X.shape[0])
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for i, (train_index, val_index) in enumerate(skf.split(X, y)):
                logger.info(f"--- Fold {i+1} started ---")

                X_train,X_val=X[train_index],X[val_index]
                y_train,y_val=y[train_index],y[val_index]

                xgb=XGBClassifier(**best_params_xgb_lvl1)
                xgb.fit(X_train,y_train)

                y_pred_proba = xgb.predict_proba(X_val)[:, 1]

                oof_preds_xgb[val_index] = y_pred_proba 

                auc_score = roc_auc_score(y_val, y_pred_proba)

                logger.info(f"Fold {i} AUC score is {auc_score:.4f}")
            
            save_numpy(oof_preds_xgb,self.model_trainer_config.xgb_level1_oof_predictions_path )
            assert_file_exists(self.model_trainer_config.xgb_level1_oof_predictions_path, "OOF lr numpy")
            log_file_size(self.model_trainer_config.xgb_level1_oof_predictions_path, "OOF lr numpy")

            overall_oof_auc = roc_auc_score(y, oof_preds_xgb)
            logger.info(f"\nOverall OOF AUC: {overall_oof_auc:.4f}")
            return oof_preds_xgb
        except Exception as e:
            logger.error("Could not generate oof xgb predictions")
            raise AITextException(e)
    
    def _generate_oof_lr_preds(self,X,y,best_params_lr_lvl1) -> np.ndarray :
        try:
            N_FOLDS = self.model_trainer_final_params_config.folds

            skf=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            oof_preds_lr = np.zeros(X.shape[0]) # Initialize OOF array with size of X

            for i,(train_idx,val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"--- Fold {i+1} started ---")

                X_train,X_val=X[train_idx],X[val_idx]
                y_train,y_val=y[train_idx],y[val_idx]

                pipe = Pipeline([
                    ("Scaler", StandardScaler()),
                    ("lr", LogisticRegression(**best_params_lr_lvl1))
                ])

                
                pipe.fit(X_train,y_train)

                y_pred_proba = pipe.predict_proba(X_val)[:, 1]
                oof_preds_lr[val_idx] = y_pred_proba 

                auc_score = roc_auc_score(y_val, y_pred_proba)
                logger.info(f"Fold {i} AUC score is {auc_score:.4f}")

            save_numpy(oof_preds_lr,self.model_trainer_config.lr_level1_oof_predictions_path )
            assert_file_exists(self.model_trainer_config.lr_level1_oof_predictions_path, "OOF lr numpy")
            log_file_size(self.model_trainer_config.lr_level1_oof_predictions_path, "OOF lr numpy")            
            overall_oof_auc = roc_auc_score(y, oof_preds_lr)
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
            save_object(xgb_lvl1,self.model_trainer_config.xgb_level1_model_path)
            self.object_storage['xgb_model']=xgb_lvl1
            

            lr_lvl1_pipeline=Pipeline([
                ('Scaler', StandardScaler()),
                ('lr', LogisticRegression(**best_params_lr_lvl1))
            ])
            lr_lvl1_pipeline.fit(X,y)
            save_object(lr_lvl1_pipeline, self.model_trainer_config.lr_level1_model_path)
            self.object_storage['lr_model']=lr_lvl1_pipeline



            X_meta = np.column_stack([oof_preds_xgb,oof_preds_lr])
            lr_lvl2_pipeline=Pipeline([
                ('Scaler', StandardScaler()),
                ('lr', LogisticRegression(**best_params_lr_lvl2))
            ])
            lr_lvl2_pipeline.fit(X_meta,y)
            save_object(lr_lvl2_pipeline, self.model_trainer_config.meta_lr_path)
            self.object_storage['meta_model']=lr_lvl2_pipeline
        except Exception as e:
            logger.error("Error occured while training")
            raise AITextException(e)




    def initiate_model_training(self):
        try:
            logger.info("Initiating model trainer")

            data = read_numpy(self.model_trainer_config.transformed_train_data_path)

            X,y =data[:,:-1], data[:,-1]

            if self.model_trainer_config.enable_tuning:
                tuner = Tuner(tuning_cfg= self.model_trainer_tuning_config)

                best_params_lr_lvl1=tuner.tune_lr_level1(X,y)
                best_params_xgb_lvl1=tuner.tune_xgb_level1(X,y)

                oof_preds_xgb=self._generate_oof_xgb_preds(X,y,best_params_xgb_lvl1)
                oof_preds_lr=self._generate_oof_lr_preds(X,y,best_params_lr_lvl1)

                X_meta = np.column_stack([oof_preds_xgb,oof_preds_lr])
                best_params_lr_lvl2=tuner.tune_lr_level2(X_meta,y)
            else:
                best_params_lr_lvl1=self.model_trainer_final_params_config.model_trainer.level1.lr
                best_params_xgb_lvl1=self.model_trainer_final_params_config.model_trainer.level1.xgb
                best_params_lr_lvl2=self.model_trainer_final_params_config.model_trainer.level2.lr

                oof_preds_xgb=self._generate_oof_xgb_preds(X,y,best_params_xgb_lvl1)
                oof_preds_lr=self._generate_oof_lr_preds(X,y,best_params_lr_lvl1)
                X_meta = np.column_stack([oof_preds_xgb,oof_preds_lr])

            self.train(X,y,oof_preds_xgb,oof_preds_lr,best_params_xgb_lvl1,best_params_lr_lvl1,best_params_lr_lvl2)
            Stacked_model=StackedModel(**self.object_storage)
            save_object(Stacked_model, self.model_trainer_config.final_model_path)
            return ModelTrainerArtifact(
                model_path= os.path.dirname(self.model_trainer_config.final_model_path)
            )
        except Exception as e:
            logger.error("Model Trainer stopped due to some error")
            raise AITextException(e)
