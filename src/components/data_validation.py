import json
import pandas as pd
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.utils.common import read_csv_file
from src.utils.logger import logger
from src.utils.exception import AITextException

class DataValidation:

    def __init__(self, cfg: DataValidationConfig) -> None:
        self.cfg = cfg
        self.report = {
            "schema_check": {},
            "missing_values_check": {},
            "allowed_values_check": {},
            "dtype_check": {}
        }

    def validate_schema(self, df) -> None:
        required = set(self.cfg.required_columns)
        present = set(df.columns)

        missing = list(required - present)

        self.report["schema_check"] = {
            "status": len(missing) == 0,
            "missing_columns": missing
        }

    def validate_missing_values(self, df) -> None:
        missing_info = df.isnull().sum().to_dict()
        status = all(v == 0 for v in missing_info.values())

        self.report["missing_values_check"] = {
            "status": status,
            "details": missing_info
        }

    def validate_allowed_values(self, df) -> None:
        invalid_rows = {}

        for column, allowed in self.cfg.allowed_values.items():

            if column not in df.columns:
                invalid_rows[column] = "Column missing"
                continue

            invalid_idx = df[~df[column].isin(allowed)].index.tolist()

            if invalid_idx:
                invalid_rows[column] = invalid_idx

        self.report["allowed_values_check"] = {
            "status": len(invalid_rows) == 0,
            "invalid_rows": invalid_rows
        }

    def validate_dtype(self, df) -> None:
        mismatches = {}

        for col, expected_dtype in self.cfg.columns_dtype.items():

            if col not in df.columns:
                mismatches[col] = {"expected": expected_dtype, "found": "Column missing"}
                continue

            actual_dtype = str(df[col].dtype)

            if expected_dtype not in actual_dtype:
                mismatches[col] = {
                    "expected": expected_dtype,
                    "found": actual_dtype
                }

        self.report["dtype_check"] = {
            "status": len(mismatches) == 0,
            "mismatched_dtypes": mismatches
        }

    def _write_report(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.report, f, indent=4)

    def _validate_df(self, df, label: str) -> None:
        logger.info(f"Running validation for: {label}")
        self.validate_schema(df)
        self.validate_missing_values(df)
        self.validate_allowed_values(df)
        self.validate_dtype(df)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation")

            df_train = read_csv_file(self.cfg.raw_train_data_path)
            df_test = read_csv_file(self.cfg.raw_test_data_path)

            self._validate_df(df_train, "Train Dataset")
            # self._validate_df(df_test, "Test Dataset")

            self._write_report(self.cfg.data_validation_report_path)

            # Fail pipeline if ANY validation fails
            for check, result in self.report.items():
                if not result["status"]:
                    logger.error(f"Validation failed: {check}")
                    raise AITextException(f"{check} failed")

            logger.info("Data validation completed successfully")

            return DataValidationArtifact(
                data_validation_report_path=self.cfg.data_validation_report_path
            )

        except Exception as e:
            logger.error("Data validation process failed")
            raise AITextException(e)
