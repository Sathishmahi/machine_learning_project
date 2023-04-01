import os,sys
import json
import pickle
from housing.entity.artifacts_entity import ModelEvaluationEntity,ModelTrainingArtifacts
from housing.config.configuration import HousingConfig
from housing.logger import logging
from housing.exception import CustomException




# class ModelEvaluation:
#     def __init__(self,model_training_artifacts:ModelTrainingArtifacts,
#                 config:HousingConfig=HousingConfig(),)->None:

#         self.model_evaluation_config=config.model_evalation_config
#         self.model_training_artifacts=model_training_artifacts

#     def to_read_json(self,json_info_file_path:str):
#         with open()