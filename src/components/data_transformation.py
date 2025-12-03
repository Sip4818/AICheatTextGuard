from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.utils.common import read_csv_file
from src.feature_generation.basic_features import BasicFeatureGenerator
from src.feature_generation.transformer_embedding import EmbeddingFeaturesGenerator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import numpy as np
import os

class DataTransformation:
    def __init__(self, cfg:DataTransformationConfig):
        self.cfg=cfg

    def split_data(self,df:pd.DataFrame):
        X,y= df.drop(columns=[self.cfg.target_column_name]),df[self.cfg.target_column_name]
        return X,y
    
    def identity_func(self,X):
        return X

    def initiate_data_transformation(self):
        train_data=read_csv_file(self.cfg.validated_data_train_path)
        X_test=read_csv_file(self.cfg.validated_data_test_path)

        X,y=self.split_data(train_data)
        y = y.to_numpy().reshape(-1, 1)

        pipeline = Pipeline([
            ('basic_features_generator', BasicFeatureGenerator()),
            ('embedding_features_generator', EmbeddingFeaturesGenerator(model_path=self.cfg.temp_model_dir)),
            ('identity', FunctionTransformer(self.identity_func))
        ])
        X=pipeline.fit_transform(X)
        np.save(self.cfg.transformed_train_data_path,np.hstack((X,y)))
        np.save(self.cfg.transformed_test_data_path,X_test)
        
        joblib.dump(pipeline, self.cfg.data_transformation_object_path)
        return DataTransformationArtifact(
            data_transformation_object_path=self.cfg.data_transformation_object_path,
            transformed_data_dir=os.path.dirname(self.cfg.transformed_train_data_path)
        )


        






 