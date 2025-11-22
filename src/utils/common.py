from src.utils.logger import logging
import yaml
from google.cloud import storage
import os
from box import ConfigBox


def read_yaml(path: str) -> ConfigBox:
    logging.info(f"Reading YAML file: {path}")
    logging.info(f"Reading YAML file: {path}")
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    return ConfigBox(content)



def download_from_gcs(bucket_name: str, source_path: str, local_path: str):
    """Downloads a blob from GCS to local."""

    # Ensure local folder exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_path)

    blob.download_to_filename(local_path)
    print(f"Downloaded {source_path} → {local_path}")
