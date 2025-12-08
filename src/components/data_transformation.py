import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from src.utils.logger import logger
from src.utils.common import (
    read_csv_file, save_numpy, assert_file_exists, 
    log_file_size, save_model
)
from src.utils.exception import AITextException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.feature_generation.basic_features import BasicFeatureGenerator
from src.feature_generation.transformer_embedding import EmbeddingFeaturesGenerator
from src.constants.constants import SEED


class DataTransformation:

    def __init__(self, cfg: DataTransformationConfig) -> None:
        self.cfg = cfg

    def split_data(self, df: pd.DataFrame):
        X = df.drop(columns=[self.cfg.target_column_name])
        y = df[self.cfg.target_column_name]

        return train_test_split(
            X, y, test_size=self.cfg.test_split_size, random_state=SEED
        )

    def _identity_func(self, X):
        return X

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation process")

            # Load validated CSVs
            df_train = read_csv_file(self.cfg.validated_data_train_path)
            df_test = read_csv_file(self.cfg.validated_data_test_path)

            # Train/val split
            X_train, X_val, y_train, y_val = self.split_data(df_train)

            y_train = y_train.to_numpy().reshape(-1, 1)
            y_val = y_val.to_numpy().reshape(-1, 1)

            # Pipeline creation
            pipeline = Pipeline([
                ('basic', BasicFeatureGenerator()),
                ('embedding', EmbeddingFeaturesGenerator(model_path=self.cfg.temp_model_dir)),
                ('identity', FunctionTransformer(self._identity_func))
            ])

            X_train = pipeline.fit_transform(X_train)
            X_val = pipeline.transform(X_val)
            X_test = pipeline.transform(df_test)

            # Save train data
            save_numpy(np.hstack((X_train, y_train)), self.cfg.transformed_train_data_path)
            assert_file_exists(self.cfg.transformed_train_data_path, "Train numpy")
            log_file_size(self.cfg.transformed_train_data_path, "Train numpy")

            # Save Validation data
            save_numpy(np.hstack((X_val, y_val)), self.cfg.transformed_val_data_path)
            assert_file_exists(self.cfg.transformed_val_data_path, "Val numpy")
            log_file_size(self.cfg.transformed_val_data_path, "Val numpy")

            # Save test data
            save_numpy(X_test, self.cfg.transformed_test_data_path)
            assert_file_exists(self.cfg.transformed_test_data_path, "Test numpy")
            log_file_size(self.cfg.transformed_test_data_path, "Test numpy")

            # Save pipeline
            save_model(pipeline, self.cfg.data_transformation_object_path)
            assert_file_exists(self.cfg.data_transformation_object_path, "Transform pipeline")
            log_file_size(self.cfg.data_transformation_object_path, "Pipeline")

            logger.info("Data transformation process completed successfully")

            return DataTransformationArtifact(
                data_transformation_object_path=self.cfg.data_transformation_object_path,
                transformed_data_dir=os.path.dirname(self.cfg.transformed_train_data_path)
            )

        except Exception as e:
            logger.error("Data Transformation process failed")
            raise AITextException(e)
