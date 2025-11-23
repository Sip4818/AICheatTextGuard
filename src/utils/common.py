from src.utils.logger import logger
from src.utils.exception import AITextException
import yaml

from google.cloud import storage
import os
from box import ConfigBox
import pandas as pd


def read_yaml(path: str) -> ConfigBox:
    logger.info(f"Reading YAML file: {path}")
    logger.info(f"Reading YAML file: {path}")
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        return ConfigBox(content)
    except Exception as e:
        logger.error("Could not read {path} file")
        raise AITextException(e)


def download_from_gcs(bucket_name: str, source_path: str, local_path: str):
    """Downloads a blob from GCS to local."""
    try:
        logger.info(f"Downloading a file from {bucket_name} bucket")
        # Ensure local folder exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_path)

        blob.download_to_filename(local_path)
        logger.info(f"Downloading successfull, file saved at {local_path}")
    except Exception as e:
        logger.error(f"Failed to Download data at: {local_path}")
        raise AITextException(e)

def read_csv_file(file_path: str) -> pd.DataFrame:
    """Read a CSV file with proper logging & error handling."""
    try:
        logger.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"CSV file read successfully: {file_path}, shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV: {file_path}")
        raise AITextException(e)