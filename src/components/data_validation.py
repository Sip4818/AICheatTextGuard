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
        self.report = {}

    def validate_schema(self, df: pd.DataFrame) -> dict:
        required = set(self.cfg.required_columns)
        present = set(df.columns)
        missing = list(required - present)
        return {
            "status": bool(len(missing) == 0), # Explicit bool for JSON serializing
            "missing_columns": missing,
        }

    def validate_missing_values(self, df: pd.DataFrame) -> dict:
        # BUG FIX: If text is an empty string "" but not NULL, isnull() misses it.
        # We check for both NaN and empty whitespace.
        missing_count = df.isnull().sum().to_dict()
        
        # Checking for empty strings in 'text' specifically
        if 'text' in df.columns:
            empty_strings = int((df['text'].astype(str).str.strip() == "").sum())
            missing_count['text_empty_string'] = empty_strings
        
        status = all(v == 0 for v in missing_count.values())
        return {
            "status": bool(status),
            "details": missing_count,
        }

    def validate_allowed_values(self, df: pd.DataFrame) -> dict:
        invalid_rows = {}
        for column, allowed in self.cfg.allowed_values.items():
            if column not in df.columns:
                continue
            
            # BUG FIX: isin() can be slow on huge sets. This is fine for [0,1].
            invalid_idx = df[~df[column].isin(allowed)].index.tolist()
            if invalid_idx:
                # Limit reported indices to 10 so the JSON report doesn't explode in size
                invalid_rows[column] = invalid_idx[:10] 
        return {
            "status": bool(len(invalid_rows) == 0),
            "invalid_rows": invalid_rows,
        }

    def validate_dtype(self, df: pd.DataFrame) -> dict:
        mismatches = {}
        for col, expected_dtype in self.cfg.columns_dtype.items():
            if col not in df.columns:
                continue
            actual_dtype = str(df[col].dtype)
            # Use 'in' because 'int64' contains 'int'
            if expected_dtype not in actual_dtype:
                mismatches[col] = {"expected": expected_dtype, "found": actual_dtype}
        return {
            "status": bool(len(mismatches) == 0),
            "mismatched_dtypes": mismatches,
        }

    def validate_data_drift_and_leakage(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
        # BUG FIX: Ensure we drop NaNs before set intersection to avoid errors
        train_texts = set(df_train['text'].dropna().astype(str).str.strip().tolist())
        test_texts = set(df_test['text'].dropna().astype(str).str.strip().tolist())
        
        overlap = train_texts.intersection(test_texts)
        status = len(overlap) == 0

        return {
            "status": bool(status),
            "overlap_count": int(len(overlap)),
            "overlap_examples": list(overlap)[:3] if not status else []
        }

    def _get_df_report(self, df: pd.DataFrame) -> dict:
        return {
            "schema_check": self.validate_schema(df),
            "missing_values_check": self.validate_missing_values(df),
            "allowed_values_check": self.validate_allowed_values(df),
            "dtype_check": self.validate_dtype(df)
        }

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation process")

            df_train = read_csv_file(self.cfg.raw_train_data_path)
            df_test = read_csv_file(self.cfg.raw_test_data_path)

            self.report["train_set"] = self._get_df_report(df_train)
            self.report["test_set"] = self._get_df_report(df_test)
            self.report["contamination_check"] = self.validate_data_drift_and_leakage(df_train, df_test)

            with open(self.cfg.data_validation_report_path, "w") as f:
                json.dump(self.report, f, indent=4)

            # Aggregate final status
            overall_status = True
            for section in ["train_set", "test_set"]:
                for check in self.report[section].values():
                    if not check["status"]:
                        overall_status = False
            
            if not self.report["contamination_check"]["status"]:
                overall_status = False

            if not overall_status:
                logger.error("Data Validation failed logic checks.")
                # Optional: You could raise Exception here to stop pipeline
                # raise AITextException("Validation Failure")

            return DataValidationArtifact(
                data_validation_report_path=self.cfg.data_validation_report_path
            )

        except Exception as e:
            logger.error(f"Data validation process failed: {str(e)}")
            raise AITextException(e)