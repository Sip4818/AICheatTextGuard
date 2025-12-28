import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from src.utils.logger import logger
from src.utils.common import (
    read_csv_file,
    save_numpy,
    assert_file_exists,
    log_file_size,
    save_object,
)
from src.utils.exception import AITextException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.feature_generation.basic_features import BasicFeatureGenerator
from src.feature_generation.transformer_embedding import EmbeddingFeaturesGenerator


class DataTransformation:
    def __init__(self, cfg: DataTransformationConfig) -> None:
        self.cfg = cfg

    def split_data(self, df: pd.DataFrame):
        df =df[self.cfg.target_column_name].copy()
        X = df.drop(columns=[self.cfg.target_column_name])
        y = df[self.cfg.target_column_name]

        return X, y

    def _identity_func(self, X):
        return X

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation process")

            # Load validated CSVs
            df_train = read_csv_file(self.cfg.validated_data_train_path)
            # df_train = df_train.sample(50)
            # Train/val split
            X, y = self.split_data(df_train)

            # Pipeline creation
            pipeline = Pipeline(
                [
                    ("basic", BasicFeatureGenerator()),
                    (
                        "embedding",
                        EmbeddingFeaturesGenerator(model_path=self.cfg.temp_model_dir),
                    ),
                    ("identity", FunctionTransformer(self._identity_func)),
                ]
            )

            X = pipeline.fit_transform(X)

            # Save train data
            save_numpy(np.column_stack((X, y)), self.cfg.transformed_train_data_path)
            assert_file_exists(self.cfg.transformed_train_data_path, "Train numpy")
            log_file_size(self.cfg.transformed_train_data_path, "Train numpy")

            # Save pipeline
            save_object(pipeline, self.cfg.data_transformation_object_path)
            assert_file_exists(
                self.cfg.data_transformation_object_path, "Transform pipeline"
            )
            log_file_size(self.cfg.data_transformation_object_path, "Pipeline")

            logger.info("Data transformation process completed successfully")

            return DataTransformationArtifact(
                data_transformation_object_path=self.cfg.data_transformation_object_path,
                transformed_data_dir=os.path.dirname(
                    self.cfg.transformed_train_data_path
                ),
            )

        except Exception as e:
            logger.error("Data Transformation process failed")
            raise AITextException(e)
