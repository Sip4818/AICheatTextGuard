from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.utils.common import read_csv_file
import pandas as pd

class DataValidation:
    def __init__(self,cfg: DataValidationConfig)-> DataValidationArtifact:
        self.cfg=cfg
        self.report = {
        "schema_check": {},
        "missing_values_check": {},
        "allowed_values_check": {},
        "dtype_check": {}
    }


    def validate_schema(self, df):
        required = set(self.cfg.required_columns)
        present  = set(df.columns)

        missing = list(required - present)

        self.report["schema_check"] = {
            "status": len(missing) == 0,
            "missing_columns": missing
        }

    
    def validate_missing_values(self, df):
        missing_info = df.isnull().sum().to_dict()

        status = all(v == 0 for v in missing_info.values())

        self.report["missing_values_check"] = {
            "status": status,
            "details": missing_info
        }

        
        

                

        
    def validate_allowed_values(self, df):
        invalid_rows = {}

        for column, allowed in self.cfg.allowed_values.items():
            invalid = df[~df[column].isin(allowed)]
            if len(invalid) > 0:
                invalid_rows[column] = invalid.index.tolist()

        self.report["allowed_values_check"] = {
            "status": len(invalid_rows) == 0,
            "invalid_rows": invalid_rows
        }

    
    def validate_dtype(self, df):
        mismatches = {}

        for col, expected_dtype in self.cfg.columns_dtype.items():
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


    def initiate_data_validation(self):
        df_train=read_csv_file(self.cfg.raw_train_data_path)
        df_test=read_csv_file(self.cfg.raw_test_data_path)

        self.validate_schema(df_train)
        self.validate_missing_values(df_train)
        self.validate_allowed_values(df_train)
        self.validate_dtype(df_train)
        with open(self.cfg.data_validation_report_path, "w") as f:
            f.write(str(self.report))

        return DataValidationArtifact(
            data_validation_report_path = self.cfg.data_validation_report_path
        )
