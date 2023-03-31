import pandas as pd
import numpy as np
import os, sys
from housing.exception import CustomException
from housing.logger import logging
from housing.constant import *
from housing.constant.experiment_models import *
from housing.constant.hyper_parameters import all_params_dict
from housing.model_training_helper.model_selection import CombineAll
from housing.config.configuration import HousingConfig
from housing.entity.artifacts_entity import DataTrainingArtifacts,FeatureEngineeringArtifacts
import yaml

class ModelTraining:
    def __init__(self,feature_engineering_artifacts:FeatureEngineeringArtifacts,
                config:HousingConfig=HousingConfig())->None:

        self.model_training_config=config.model_transformation_config
        self.feature_engineering_artifacts=feature_engineering_artifacts

    def initiate_model_training(self,):
        trained_model_dir=self.model_training_config.trained_model_dir
        base_accuracy=self.model_training_config.base_accuracy
        model_config_file_path=self.model_training_config.model_config_file_path
        with open(model_config_file_path,'r') as yaml_file:
            all_models_name_list=yaml.safe_load(yaml_file).get(MODEL_TRAINING_DICT_KEY).get(MODEL_TRAINING_MODEL_NAMES_LIST)
        
        train_file_path=feature_engineering_artifacts.train_file_path
        test_file_path=feature_engineering_artifacts.train_file_path
        target_cloumn_name=
        combine=CombineAll(all_model_names_list=all_models_name_list)
        combine.to_return_best_model(df=, target_col_name, 
        to_stote_model_path, test_data, json_training_info_file_path, cluster_file_path)


        