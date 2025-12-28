from operator import index
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


def is_yaml_content_empty(file_path: str) -> bool:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data is None or data == {}


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
            raise AITextException(
                f"Invalid YAML structure in {path}: Must be a dictionary"
            )

        logger.info(f"YAML loaded successfully: {path}")
        return ConfigBox(content)

    except yaml.YAMLError as e:
        logger.error(f"YAML syntax error in: {path}")
        raise AITextException(e)

    except Exception as e:
        logger.error(f"Could not read YAML file: {path}")
        raise AITextException(e)


import yaml
from pathlib import Path


def write_yaml(data: dict, file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def create_dir(path: str, name: str) -> None:
    """Create directory"""
    try:
        os.makedirs(path, exist_ok=True)
        logger.info(f"{name} directory created at: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {name}: {path}")
        raise AITextException(e)


def upload_to_gcs(
    bucket_name: str,
    source_path: str,
    destination_path: str,
    overwrite: bool = True,
    timeout: int = 600,
) -> None:
    """
    Upload a local file to Google Cloud Storage.

    Args:
        bucket_name (str): Name of the GCS bucket
        source_path (str): Local file path to upload
        destination_path (str): GCS object path (inside bucket)
        overwrite (bool): Whether to overwrite if object already exists

    Raises:
        AITextException: If upload fails
    """
    try:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_path)

        if blob.exists() and not overwrite:
            logger.info(
                f"GCS object already exists, skipping upload: "
                f"gs://{bucket_name}/{destination_path}"
            )
            return

        logger.info(
            f"Uploading file to GCS: "
            f"{source_path} → gs://{bucket_name}/{destination_path}"
        )

        blob.upload_from_filename(source_path, timeout=timeout)

        logger.info(
            f"Upload completed: gs://{bucket_name}/{destination_path}"
        )

    except Exception as e:
        logger.error("Failed to upload file to GCS")
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

def save_csv(df: pd.DataFrame, path: str) -> None:
    try:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        df.to_csv(path, index= False)

        logger.info(f"CSV file saved at {path}")
    except Exception as e:
        logger.error(f"Failed to save csv at {path}")
        raise AITextException(e)


def read_object(model_path: str) -> object:
    """read object or a model as pkl"""
    try:
        if not os.path.exists(model_path):
            raise AITextException("Model does not exist")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Could not read the model {model_path}")
        raise AITextException(e)


def save_object(model: object, model_path: str) -> None:
    """save object or a model as pkl"""
    try:
        dir_path = os.path.dirname(model_path)
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


def save_numpy(array: np.ndarray, path: str) -> None:
    try:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        np.save(path, array)

        logger.info(f"numpy array saved at {path}")
    except Exception as e:
        logger.error(f"Failed to save numpy array at {path}")
        raise AITextException(e)


def extract_params(source: dict, keys: list[str]) -> dict:
    clean = {}
    for k in keys:
        v = source[k]
        if isinstance(v, (int, float, str, bool)):
            clean[k] = v
        else:
            clean[k] = float(v)
    return clean


def to_dict(obj):
    """
    Recursively convert ConfigBox / objects to pure Python dict
    """
    if isinstance(obj, ConfigBox):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(v) for v in obj]
    else:
        return obj
