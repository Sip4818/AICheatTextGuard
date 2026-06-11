import os
from config.configuration import ConfigurationManager
from config.training_pipeline_config import TrainingPipelineConfig


# Config Manager test cases
def test_config_init():

    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )

    assert config_manager is not None


# Test data ingestion config retrieval
def test_get_data_ingestion_config():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )

    data_ingestion_config = config_manager.get_data_ingestion_config()

    assert data_ingestion_config is not None
    assert hasattr(data_ingestion_config, "cloud_data_path")
    assert hasattr(data_ingestion_config, "bucket_name")
    assert hasattr(data_ingestion_config, "local_data_path")
    assert hasattr(data_ingestion_config, "local_train_path")
    assert hasattr(data_ingestion_config, "local_test_path")
    assert hasattr(data_ingestion_config, "test_split_size")
    assert hasattr(data_ingestion_config, "target_column_name")
    assert hasattr(data_ingestion_config, "required_columns")
    assert hasattr(data_ingestion_config, "to_download_data")

    assert isinstance(data_ingestion_config.cloud_data_path, str)
    assert isinstance(data_ingestion_config.bucket_name, str)
    assert isinstance(data_ingestion_config.local_data_path, str)
    assert isinstance(data_ingestion_config.local_train_path, str)
    assert isinstance(data_ingestion_config.local_test_path, str)
    assert isinstance(data_ingestion_config.test_split_size, float)
    assert isinstance(data_ingestion_config.target_column_name, str)
    assert isinstance(data_ingestion_config.required_columns, list)
    assert isinstance(data_ingestion_config.to_download_data, bool)

    assert 0 < data_ingestion_config.test_split_size < 1
    assert len(data_ingestion_config.required_columns) > 0


# Test if data ingestion config paths are created
def test_data_ingestion_config_paths():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )
    config = config_manager.get_data_ingestion_config()

    assert os.path.exists(os.path.dirname(config.local_train_path))


# Test data validation config retrieval
def test_get_data_validation_config():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )

    data_validation_config = config_manager.get_data_validation_config()

    assert data_validation_config is not None

    assert hasattr(data_validation_config, "raw_train_data_path")
    assert hasattr(data_validation_config, "raw_test_data_path")
    assert hasattr(data_validation_config, "data_validation_report_path")
    assert hasattr(data_validation_config, "required_columns")
    assert hasattr(data_validation_config, "columns_dtype")
    assert hasattr(data_validation_config, "allowed_values")

    assert isinstance(data_validation_config.raw_train_data_path, str)
    assert isinstance(data_validation_config.raw_test_data_path, str)
    assert isinstance(data_validation_config.data_validation_report_path, str)
    assert isinstance(data_validation_config.required_columns, list)
    assert isinstance(data_validation_config.columns_dtype, dict)
    assert isinstance(data_validation_config.allowed_values, dict)

    assert len(data_validation_config.required_columns) > 0
    assert len(data_validation_config.columns_dtype) > 0


# Check if data validation config paths are created
def test_data_validation_config_paths():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )
    config = config_manager.get_data_validation_config()

    assert os.path.exists(os.path.dirname(config.data_validation_report_path))


# Test data transformation config retrieval
def test_get_data_transformation_config():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )

    data_transformation_config = config_manager.get_data_transformation_config()
    assert data_transformation_config is not None
    assert hasattr(data_transformation_config, "validated_data_train_path")
    assert hasattr(data_transformation_config, "transformed_train_data_path")
    assert hasattr(data_transformation_config, "data_transformation_object_path")
    assert hasattr(data_transformation_config, "temp_model_dir")
    assert hasattr(data_transformation_config, "target_column_name")
    assert hasattr(data_transformation_config, "test_split_size")
    assert hasattr(data_transformation_config, "required_columns")

    assert isinstance(data_transformation_config.validated_data_train_path, str)
    assert isinstance(data_transformation_config.transformed_train_data_path, str)
    assert isinstance(data_transformation_config.data_transformation_object_path, str)
    assert isinstance(data_transformation_config.temp_model_dir, str)
    assert isinstance(data_transformation_config.target_column_name, str)
    assert isinstance(data_transformation_config.test_split_size, float)
    assert isinstance(data_transformation_config.required_columns, list)

    assert 0 < data_transformation_config.test_split_size < 1
    assert len(data_transformation_config.required_columns) > 0


# Check if data transformation config paths are created
def test_data_transformation_config_paths():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )
    config = config_manager.get_data_transformation_config()

    assert os.path.exists(os.path.dirname(config.transformed_train_data_path))
    assert os.path.exists(os.path.dirname(config.data_transformation_object_path))


# Test model trainer config retrieval
def test_get_model_trainer_config():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )

    model_trainer_config = config_manager.get_model_trainer_config()

    assert model_trainer_config is not None
    assert hasattr(model_trainer_config, "transformed_train_data_path")
    assert hasattr(model_trainer_config, "preprocessing_object_path")
    assert hasattr(model_trainer_config, "lr_level1_model_path")
    assert hasattr(model_trainer_config, "xgb_level1_model_path")
    assert hasattr(model_trainer_config, "meta_lr_path")
    assert hasattr(model_trainer_config, "enable_tuning")
    assert hasattr(model_trainer_config, "final_model_path")
    assert hasattr(model_trainer_config, "lr_level1_oof_predictions_path")
    assert hasattr(model_trainer_config, "xgb_level1_oof_predictions_path")
    assert hasattr(model_trainer_config, "folds")

    assert isinstance(model_trainer_config.transformed_train_data_path, str)
    assert isinstance(model_trainer_config.preprocessing_object_path, str)
    assert isinstance(model_trainer_config.lr_level1_model_path, str)
    assert isinstance(model_trainer_config.xgb_level1_model_path, str)
    assert isinstance(model_trainer_config.meta_lr_path, str)
    assert isinstance(model_trainer_config.enable_tuning, bool)
    assert isinstance(model_trainer_config.final_model_path, str)
    assert isinstance(model_trainer_config.lr_level1_oof_predictions_path, str)
    assert isinstance(model_trainer_config.xgb_level1_oof_predictions_path, str)
    assert isinstance(model_trainer_config.folds, int)
    assert model_trainer_config.folds > 1
    assert model_trainer_config.folds < 20


# Check if model trainer config paths are created
def test_model_trainer_config_paths():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )
    config = config_manager.get_model_trainer_config()

    assert os.path.exists(os.path.dirname(config.lr_level1_model_path))
    assert os.path.exists(os.path.dirname(config.lr_level1_oof_predictions_path))
    assert os.path.exists(os.path.dirname(config.final_model_path))


# Test model trainer tuning config retrieval
def test_get_model_trainer_tuning_config():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )

    tuning_config = config_manager.get_model_trainer_tuning_config()
    assert tuning_config is not None
    assert hasattr(tuning_config, "level1")
    assert hasattr(tuning_config, "level2")
    assert hasattr(tuning_config, "n_trials")


# Tests for final model evaluation config retrieval
def test_get_model_trainer_final_params_config():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )

    final_model_evaluation_config = (
        config_manager.get_model_trainer_final_params_config()
    )
    assert final_model_evaluation_config is not None
    assert hasattr(final_model_evaluation_config, "level1")
    assert hasattr(final_model_evaluation_config, "level2")


# Tests for model evaluation config retrieval
def test_get_model_evaluation_config():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )
    config = config_manager.get_model_evaluation_config()

    assert config is not None
    assert hasattr(config, "raw_test_data_path")
    assert hasattr(config, "final_model_path")
    assert hasattr(config, "target_column_name")
    assert hasattr(config, "model_evaluation_artifact_file_path")
    assert hasattr(config, "plot_file_path")
    assert hasattr(config, "metrices")
    assert hasattr(config, "push_model_to_gcs")
    assert hasattr(config, "gcs_bucket_name")

    assert isinstance(config.raw_test_data_path, str)
    assert isinstance(config.final_model_path, str)
    assert isinstance(config.target_column_name, str)
    assert isinstance(config.model_evaluation_artifact_file_path, str)
    assert isinstance(config.plot_file_path, str)
    assert isinstance(config.metrices, list)
    assert isinstance(config.push_model_to_gcs, bool)
    assert isinstance(config.gcs_bucket_name, str)
    assert len(config.metrices) > 0
    assert config.push_model_to_gcs in [True, False]


# Check if model evaluation config paths are created
def test_model_evaluation_config_paths():
    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(
        training_pipeline_config=training_pipeline_config
    )
    config = config_manager.get_model_evaluation_config()

    assert os.path.exists(os.path.dirname(config.model_evaluation_artifact_file_path))
