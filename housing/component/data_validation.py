from housing.config.configuration import HousingConfig
from housing.logger import logging
from housing.exception import CustomException
from housing.utils.util import read_yaml
from housing.entity.artifacts_entity import DataInjectionArtifacts
from housing.constant import *
import os,sys
import pandas as pd
class DataValidation:
    def __init__(self, config: HousingConfig,data_injection_artifacts:DataInjectionArtifacts):
        try:
            self.config = config
            self.data_injection_artifacts=data_injection_artifacts
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e   
    def read_yaml_and_return_content(self,key:str)->dict.values:
        try:
            data_validation_schema_file_path=self.config.schema_file_path
            if os.path.exists(data_validation_schema_file_path):
                file_content=read_yaml(file_path=data_validation_schema_file_path)
                return file_content.get(key)
            return None
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e
    def validate_data_columns_and_dtypes(self)->bool:
        try:
            is_statified=False
            train_file_path=self.data_injection_artifacts.train_file_path
            test_file_path=self.data_injection_artifacts.test_file_path
            if all(  [os.path.join(train_file_path),os.path.join(test_file_path)]  ):        
                all_columns_dtypes=self.read_yaml_and_return_content(key=COLUMNS_KEY)         
                all_columns=list(all_columns_dtypes.keys())
                train_df=pd.read_csv(train_file_path,usecols=all_columns)
                test_df=pd.read_csv(test_file_path,usecols=all_columns)
                if len(all_columns)==train_df.shape[1]==test_df.shape[1]:
                    check_len=[]
                    for col in train_df.columns:
                        schema_dtype=all_columns_dtypes.get(col)
                        df_dtype=train_df[col].dtypes
                        if (col in all_columns) and ((schema_dtype=='O' and df_dtype=='category') or (schema_dtype=='category' and df_dtype=='O') or (schema_dtype==df_dtype)):
                            check_len.append(True)
                    if len(check_len)==train_df.shape[1]:
                        is_statified = True
                return is_statified
            return False
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e


    def validate_uniques_of_columns(self)->bool:
        try:
            train_file_path=self.data_injection_artifacts.train_file_path
            test_file_path=self.data_injection_artifacts.test_file_path
            if all( [os.path.exists(train_file_path),os.path.exists(test_file_path)] ): 
                train_df=pd.read_csv(train_file_path)
                test_df=pd.read_csv(test_file_path)

                uniques=self.read_yaml_and_return_content(DOMAIN_VALUE_KEY).get(INSIDE_DOMAIN_VALUE_KEY)
                data_uniques=train_df[INSIDE_DOMAIN_VALUE_KEY].unique()
                is_statified=False
                if len(data_uniques)==len(uniques) and len([True for uni in data_uniques if uni in uniques])==len(uniques):
                    is_statified=True
                return is_statified
            return False
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e

    def initiate_data_validation(self)->bool:
        
        try:
            return all([self.validate_uniques_of_columns(),self.validate_data_columns_and_dtypes()])
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e
    
    

