from config.configuration import ConfigurationManager
from config.training_pipeline_config import TrainingPipelineConfig

#Config Manager test cases
def test_config():

    training_pipeline_config = TrainingPipelineConfig()
    config_manager = ConfigurationManager(training_pipeline_config=training_pipeline_config)

    assert config_manager is not None




