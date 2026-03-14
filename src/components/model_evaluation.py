from src.utils.common import read_object, read_csv_file, upload_to_gcs
from src.utils.exception import AITextException
from src.utils.logger import logger
from src.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import roc_auc_score
import pandas as pd
import json
from typing import Any
from dotenv import load_dotenv
load_dotenv()

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


    def push_model_to_gcs(self):

        upload_to_gcs(
            bucket_name=self.cfg.gcs_bucket_name,
            source_path=self.cfg.final_model_path,
            destination_path=self.cfg.final_model_path,
            overwrite=True,
        )
    def initiate_model_evaluation(self):
        try:
            logger.info("Starting model evaluation")
            model = read_object(self.cfg.final_model_path)

            test_data = read_csv_file(self.cfg.raw_test_data_path)
            test_data = test_data.head(10)
            X, y = self._split_data(test_data)

            y_pred = model.predict_proba(X)[:, 1]

            score = roc_auc_score(y, y_pred)
            content=f"Final ROC AUC Score of the model is {score}"
            self._write_report(self.cfg.model_evaluation_artifact_file_path, content=content)

            if self.cfg.push_model_to_gcs:
                self.push_model_to_gcs()
            logger.info(f"Final ROC AUC Score of the model is {score}")
        except Exception as e:
            logger.error("Couldn't evaluate the final model")
            raise AITextException(e)
