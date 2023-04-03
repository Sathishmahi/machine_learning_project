import os
import sys

import numpy as np
import pandas as pd
import yaml

from housing.config.configuration import HousingConfig
from housing.constant import *
from housing.constant.experiment_models import *
from housing.constant.hyper_parameters import all_params_dict
from housing.entity.artifacts_entity import (FeatureEngineeringArtifacts,
                                             ModelTrainingArtifacts)
from housing.exception import CustomException
from housing.logger import logging
from housing.model_training_helper.model_selection import CombineAll
from housing.utils import util


class ModelTraining:
    def __init__(self,feature_engineering_artifacts:FeatureEngineeringArtifacts,
                config:HousingConfig=HousingConfig())->None:
        """
        ModelTraining to train our model

        Args:
            feature_engineering_artifacts (FeatureEngineeringArtifacts): all data files path
            config (HousingConfig, optional): all config class. Defaults to HousingConfig().

        Raises:
            CustomException: 
        """
        
        try:
            self.model_training_config=config.model_training_config
            self.schema_file_path=config.data_validation_config.schema_file_path
            self.feature_engineering_artifacts=feature_engineering_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    def initiate_model_training(self)->ModelTrainingArtifacts:
        """
        initiate_model_training to combine all functions

        Raises:
            CustomException: 

        Returns:
            ModelTrainingArtifacts: all trained (cluster and ml) model path
        """
        try:
            trained_model_dir=self.model_training_config.trained_model_dir
            base_accuracy=self.model_training_config.base_accuracy
            model_config_file_path=self.model_training_config.model_config_file_path
            model_info_json_file_path=self.model_training_config.model_info_jaon_file_path
            cluster_model_file_path=self.model_training_config.cluster_model_file_path
            overall_model_info_json_file_path=self.model_training_config.overall_models_info_json_path
            
            with open(model_config_file_path,'r') as yaml_file:
                all_models_name_list=yaml.safe_load(yaml_file).get(MODEL_TRAINING_DICT_KEY).get(MODEL_TRAINING_MODEL_NAMES_LIST)
            
            
            for path in [trained_model_dir,os.path.dirname(cluster_model_file_path)]:
                os.makedirs(path,exist_ok=True)
            
            train_file_path=self.feature_engineering_artifacts.transformed_train_file_path
            test_file_path=self.feature_engineering_artifacts.transformed_test_file_path
            target_cloumn_name=OUT_COME_COLUMN_NAME
            combine=CombineAll(all_model_names_list=all_models_name_list)

            model_training_artifacts=combine.to_return_best_model(df=pd.read_parquet(train_file_path).iloc[:1000,:], target_col_name=target_cloumn_name, 
            to_stote_model_path=trained_model_dir, test_data=pd.read_parquet(test_file_path).iloc[:250,:], 
            json_training_info_file_path=model_info_json_file_path, n_clusters=2,cluster_file_path=cluster_model_file_path,overall_model_info_json_file_path=overall_model_info_json_file_path)
            return model_training_artifacts

        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
        