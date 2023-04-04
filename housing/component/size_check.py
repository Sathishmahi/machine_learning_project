import os
import sys

import numpy as np
import pandas as pd
import yaml

from housing.constant import *
from housing.entity.artifacts_entity import (DataInjectionArtifacts,
                                             DataValidationArtifacts)
from housing.exception import CustomException
from housing.logger import logging

# {

#     'float16_max':65500.0,
#     'float16_min':6.10 *10**-5,

#     'float32_max':3.4028237 * 10**38,
#     'float32_min':1.175494 * 10**-38,

#     'category_thersold':15


# }

    
class CheckDataSize:

    def __init__(self,data_injection_artifacts:DataInjectionArtifacts):
        """
        CheckDataSize to check the data size and assign to possible dtypes help of size checking yaml

        Args:
            data_injection_artifacts (DataInjectionArtifacts): all downloaded data file path

        Raises:
            CustomException: 
        """
        try:
            self.data_injection_artifacts=data_injection_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)


    def to_read_yaml(self,yaml_file_path:str=SIZE_CHECKING_YAML_FILE_PATH)->dict:
        """
        to_read_yaml to read yaml file return file content

        Args:
            yaml_file_path (str, optional): size checking yaml file path. Defaults to SIZE_CHECKING_YAML_FILE_PATH.

        Raises:
            FileNotFoundError: 
            CustomException: 

        Returns:
            dict: to return file content
        """
        try:
            if not os.path.exists(yaml_file_path):
                raise FileNotFoundError(f'yaml file not fount {yaml_file_path} ')
            with open(yaml_file_path,'r') as yaml_file:
                yaml_content=yaml.safe_load(yaml_file)
            return yaml_content
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def to_check_range(self)->None:
        """
        to_check_range to check the all columns range (min and max) and assign possible dtypes

        Note: float16 not applicable because of parquet file format not accept float16

        Raises:
            CustomException: 

        Returns:
            None
        """
        try:

            train_file_path=self.data_injection_artifacts.train_file_path
            test_file_path=self.data_injection_artifacts.test_file_path

            train_df=pd.read_parquet(train_file_path)
            test_df=pd.read_parquet(test_file_path)
            content=self.to_read_yaml()
            for col,val in zip(train_df.dtypes.index,train_df.dtypes.values):

                if train_df[col].dtypes not in ['category','O'] :
                    # if train_df[col].min()>=content.get(FLOAT16_MIN_KEY) and train_df[col].max()<=content.get(FLOAT16_MAX_KEY):
                    #     train_df[col]=train_df[col].astype('float16')
                    #     print(f'column name ',col)
                    if np.min(train_df[col])>=content.get(FLOAT32_MIN_KEY) and np.max(train_df[col])<=content.get(FLOAT32_MAX_KEY):
                        train_df[col]=train_df[col].astype('float32')
                    else:
                        train_df[col]=train_df[col].astype('float64')
                else:
                    if train_df[col].nunique()<=content.get('category_thersold'):
                        train_df[col]=train_df[col].astype('category')

            test_df=test_df.astype(  {col:train_df.dtypes[col]  for col in test_df.columns}  )
            for df,path in zip([train_df,test_df],[train_file_path,test_file_path]):
                df.to_parquet(path,index=False,engine='pyarrow',compression='gzip')
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def initiate_size_check(self)->None:
        """
        initiate_size_check to combine all function

        Raises:
            CustomException: 

        Returns:
            None
        """
        try:
            self.to_check_range()
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)