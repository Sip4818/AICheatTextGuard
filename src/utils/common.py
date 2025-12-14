import logging
import os
from pickle import NONE
import yaml
import joblib
import pandas as pd
from box import ConfigBox
from google.cloud import storage
import numpy as np

from src.utils.logger import logger
from src.utils.exception import AITextException


def assert_file_exists(path: str, label: str = "File") -> None:
    if not os.path.exists(path):
        raise AITextException(f"{label} does not exist at path: {path}")

def read_yaml(path: str) -> ConfigBox:
    """Read yaml file"""
    logger.info(f"Reading YAML file: {path}")

    try:
        if not os.path.exists(path):
            raise AITextException(f"YAML file does not exist: {path}")

        with open(path, "r") as f:
            content = yaml.safe_load(f)

        if content is None:
            raise AITextException(f"YAML file is empty: {path}")

        if not isinstance(content, dict):
            raise AITextException(f"Invalid YAML structure in {path}: Must be a dictionary")

        logger.info(f"YAML loaded successfully: {path}")
        return ConfigBox(content)

    except yaml.YAMLError as e:
        logger.error(f"YAML syntax error in: {path}")
        raise AITextException(e)

    except Exception as e:
        logger.error(f"Could not read YAML file: {path}")
        raise AITextException(e)


def create_dir(path: str, name: str) -> None:
    """ Create directory"""
    try:
        os.makedirs(path, exist_ok=True)
        logger.info(f"{name} directory created at: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {name}: {path}")
        raise AITextException(e)


def download_from_gcs(bucket_name: str, source_path: str, local_path: str) -> None:
    """Downloads a blob from GCS to local."""
    try:
        logger.info(f"Downloading from GCS bucket={bucket_name}, blob={source_path}")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if not bucket.exists(client):
            raise AITextException(f"GCS bucket does not exist: {bucket_name}")

        blob = bucket.blob(source_path)

        if not blob.exists(client):
            raise AITextException(f"File does not exist in bucket: {source_path}")

        blob.download_to_filename(local_path)

        logger.info(f"Download successful → saved at: {local_path}")

    except Exception as e:
        logger.error(f"Failed to download from GCS → {bucket_name}/{source_path}")
        raise AITextException(e)


def read_csv_file(file_path: str) -> pd.DataFrame:
    """Read csv file"""
    try:
        logger.info(f"Reading CSV file: {file_path}")

        if not file_path.endswith(".csv"):
            raise AITextException(f"Not a CSV file: {file_path}")

        df = pd.read_csv(file_path)

        logger.info(f"CSV loaded successfully: {file_path}, shape={df.shape}")

        return df

    except Exception as e:
        logger.error(f"Failed to read CSV: {file_path}")
        raise AITextException(e)


def read_object(model_path: str) -> object:
    """read object or a model as pkl"""
    try:
        if not os.path.exists(model_path):
            raise AITextException('Model does not exist')
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f'Could not read the model {model_path}')
        raise AITextException(e)


def save_object(model: object, model_path: str) -> None:
    """ save object or a model as pkl"""
    try:
        dir_path=os.path.dirname(model_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved at {model_path}")

    except Exception as e:
        logger.error(f"Failed to save model at {model_path}")
        raise AITextException(e)


def log_file_size(path: str, label: str = "File") -> None:
    """Loggin file size"""
    try:
        size_kb = os.path.getsize(path) / 1024
        logger.info(f"{label} size: {size_kb:.2f} KB")
    except Exception as e:
        logger.error(f"Failed to get file size for {path}")
        raise AITextException(e)




def read_numpy(path: str) -> np.ndarray:
    try:
        if not path.endswith(".npy"):
            raise AITextException(f"Not a npy file: {path}")
        return np.load(path)
    except Exception as e:
        logger.error(f"Could not read numpy array at {path}")
        raise AITextException(e)


def save_numpy(array: np.ndarray, path: str ) -> None:
    try:
        dir_path=os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        np.save(path, array)

        logger.info(f"numpy array saved at {path}")
    except Exception as e:
        logger.error(f"Failed to save numpy array at {path}")
        raise AITextException(e)

    