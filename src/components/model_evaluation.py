from src.utils.common import read_object, read_csv_file, upload_to_gcs
from src.utils.exception import AITextException
from src.utils.logger import logger
from src.entity.config_entity import ModelEvaluationConfig
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
            X, y = self._split_data(test_data)

            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)

            auc_score = roc_auc_score(y, y_pred_proba)
            precision = precision_score(y, y_pred_binary)
            recall = recall_score(y, y_pred_binary)
            f1 = f1_score(y, y_pred_binary)
            accuracy = accuracy_score(y, y_pred_binary)
            cm = confusion_matrix(y, y_pred_binary)
            # plot
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")

            # save
            plt.savefig(self.cfg.plot_file_path, dpi=300, bbox_inches="tight")
            plt.close()
            metrics = {
                "roc_auc": round(auc_score, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(accuracy, 4)
            }


            # After getting your predictions (y_pred) and true labels (y_test)
            # report = classification_report(y, y_pred_binary, output_dict=True)

            # Save as JSON for easy reading/tracking


            self._write_report(self.cfg.model_evaluation_artifact_file_path, content=metrics)
            logger.info(metrics)

            if self.cfg.push_model_to_gcs:
                self.push_model_to_gcs()
        except Exception as e:
            logger.error("Couldn't evaluate the final model")
            raise AITextException(e)
