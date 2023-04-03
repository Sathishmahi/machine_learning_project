import os
import sys

import pandas as pd

from housing.config.configuration import HousingConfig
from housing.constant import *
from housing.entity.artifacts_entity import (DataInjectionArtifacts,
                                             DataValidationArtifacts)
from housing.exception import CustomException
from housing.logger import logging
from housing.utils.util import read_yaml


class DataValidation:

    def __init__(
        self, config: HousingConfig, data_injection_artifacts: DataInjectionArtifacts
    ):
        """
        DataValidation help us to validate data using schema config

        Args:
            config (HousingConfig): config class 
            data_injection_artifacts (DataInjectionArtifacts): data injection artifacts(data injection all file path)

        Raises:
            CustomException: 
        """
        try:
            self.config = config
            self.data_injection_artifacts = data_injection_artifacts
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def read_yaml_and_return_content(self, key: str) -> dict.values:
        """
        read_yaml_and_return_content to read a yaml file and return  content
        Args:
            key (str): key for yaml content

        Raises:
            CustomException: 

        Returns:
            dict.values: to return dict value by given key
        """
        try:
            data_validation_schema_file_path = self.config.schema_file_path
            if os.path.exists(data_validation_schema_file_path):
                file_content = read_yaml(file_path=data_validation_schema_file_path)
                return file_content.get(key)
            return None
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def validate_data_columns_and_dtypes(self) -> bool:
        """
        validate_data_columns_and_dtypes to validate a data columns and dtypes by using schema config
        Raises:
            CustomException: 

        Returns:
            bool: if all are match (downloaded data and schema config) then return True 
            else to return false
        """
        try:
            is_statified = False
            train_file_path = self.data_injection_artifacts.train_file_path
            test_file_path = self.data_injection_artifacts.test_file_path
            if all([os.path.join(train_file_path), os.path.join(test_file_path)]):
                all_columns_dtypes = self.read_yaml_and_return_content(key=COLUMNS_KEY)
                all_columns = list(all_columns_dtypes.keys())
                train_df = pd.read_parquet(train_file_path)
                test_df = pd.read_parquet(test_file_path)
                if len(all_columns) == train_df.shape[1] == test_df.shape[1]:
                    check_len = []
                    for col in train_df.columns:
                        schema_dtype = all_columns_dtypes.get(col)
                        df_dtype = train_df[col].dtypes
                        if (col in all_columns) and (
                            (schema_dtype == "O" and df_dtype == "category")
                            or (schema_dtype == "category" and df_dtype == "O")
                            or (schema_dtype == df_dtype)
                        ):
                            check_len.append(True)
                    if len(check_len) == train_df.shape[1]:
                        is_statified = True
                return is_statified
            return False

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def validate_uniques_of_columns(self) -> bool:
        """
        validate_uniques_of_columns to validate unique vlaues of the categorical columns 

        Raises:
            CustomException: 

        Returns:
            bool: if all uniques match to schema config then to return True
            else return False
        """
        try:
            train_file_path = self.data_injection_artifacts.train_file_path
            test_file_path = self.data_injection_artifacts.test_file_path
            if all([os.path.exists(train_file_path), os.path.exists(test_file_path)]):
                train_df = pd.read_parquet(train_file_path)
                test_df = pd.read_parquet(test_file_path)

                uniques = self.read_yaml_and_return_content(DOMAIN_VALUE_KEY).get(
                    INSIDE_DOMAIN_VALUE_KEY
                )
                data_uniques = train_df[INSIDE_DOMAIN_VALUE_KEY].unique()
                is_statified = False
                if len(data_uniques) == len(uniques) and len(
                    [True for uni in data_uniques if uni in uniques]
                ) == len(uniques):
                    is_statified = True
                return is_statified
            return False
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifacts:
        """
        initiate_data_validation to combine all functions  

        Raises:
            CustomException: 

        Returns
            DataValidationArtifacts: to return all DataValidationArtifacts
        """
        try:
            json_report_file_path = self.config.report_file_path
            data_validation_artifacts = DataValidationArtifacts(
                all(
                    [
                        self.validate_uniques_of_columns(),
                        self.validate_data_columns_and_dtypes(),
                    ]
                ),
                json_report_file_path,
            )
            return data_validation_artifacts
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e
