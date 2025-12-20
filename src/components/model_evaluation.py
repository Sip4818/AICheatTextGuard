from os import write
from src.utils.common import read_object, read_csv_file
from src.utils.exception import AITextException
from src.utils.logger import logger
from src.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import roc_auc_score
import pandas as pd
import json
from typing import Any

class ModelEvaluation:
    def __init__(self, cfg: ModelEvaluationConfig):
        self.cfg = cfg
    def _write_report(self, path: str, content: Any) -> None:
        with open(path, "w") as f:
            json.dump(content, f, indent=4)

    def _split_data(self, df: pd.DataFrame):
        X = df.drop(columns=[self.cfg.target_column_name])
        y = df[self.cfg.target_column_name]
        return X, y

    def initiate_model_evaluation(self):
        try:
            logger.info("Starting model evaluation")
            model = read_object(self.cfg.final_model_path)

            test_data = read_csv_file(self.cfg.raw_test_data_path)

            X, y = self._split_data(test_data)

            y_pred = model.predict_proba(X)[:, 1]

            score = roc_auc_score(y, y_pred)
            content=f"Final ROC AUC Score of the model is {score}"
            self._write_report(self.cfg.model_evaluation_artifact_file_path, content=content)

            logger.info(f"Final ROC AUC Score of the model is {score}")
            logger.info("model evaluation stage complete")
        except Exception as e:
            logger.error("Couldn't evaluate the final model")
            raise AITextException(e)
