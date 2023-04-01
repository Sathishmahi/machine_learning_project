import pandas as pd
import numpy as np
import os, sys
import yaml
from housing.exception import CustomException
from housing.logger import logging
from housing.constant import *
from housing.utils import util
from housing.constant.experiment_models import *
from housing.constant.hyper_parameters import all_params_dict
from housing.model_training_helper.model_selection import CombineAll
from housing.config.configuration import HousingConfig
from housing.entity.artifacts_entity import ModelTrainingArtifacts,FeatureEngineeringArtifacts


class ModelTraining:
    def __init__(self,feature_engineering_artifacts:FeatureEngineeringArtifacts,
                config:HousingConfig=HousingConfig())->None:
        
        try:
            self.model_training_config=config.model_training_config
            self.schema_file_path=config.data_validation_config.schema_file_path
            self.feature_engineering_artifacts=feature_engineering_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    def initiate_model_training(self)->ModelTrainingArtifacts:
        try:
            trained_model_dir=self.model_training_config.trained_model_dir
            base_accuracy=self.model_training_config.base_accuracy
            model_config_file_path=self.model_training_config.model_config_file_path
            model_info_json_file_path=self.model_training_config.model_info_jaon_file_path
            cluster_model_file_path=self.model_training_config.cluster_model_file_path
            schema_file_path='/config/workspace/config/schema.yaml'
            with open(model_config_file_path,'r') as yaml_file:
                all_models_name_list=yaml.safe_load(yaml_file).get(MODEL_TRAINING_DICT_KEY).get(MODEL_TRAINING_MODEL_NAMES_LIST)
            
            
            for path in [trained_model_dir,os.path.dirname(cluster_model_file_path)]:
                os.makedirs(path,exist_ok=True)
            train_file_path=self.feature_engineering_artifacts.transformed_train_file_path
            test_file_path=self.feature_engineering_artifacts.transformed_test_file_path
            target_cloumn_name=util.read_yaml(file_path=schema_file_path).get(TARGET_COLUMN_KEY)
            combine=CombineAll(all_model_names_list=all_models_name_list)

            model_training_artifacts=combine.to_return_best_model(df=pd.read_csv(train_file_path).iloc[:1000,:], target_col_name=target_cloumn_name, 
            to_stote_model_path=trained_model_dir, test_data=pd.read_csv(test_file_path).iloc[:250,:], 
            json_training_info_file_path=model_info_json_file_path, cluster_file_path=cluster_model_file_path)
            return model_training_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
        