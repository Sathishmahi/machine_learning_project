import os
import sys
from housing.config.configuration import HousingConfig
from housing.component.data_injection import DataInjection
from housing.component.data_validation import DataValidation
from housing.logger import logging

from housing.exception import CustomException
from housing.entity.artifacts_entity import DataInjectionArtifacts
from housing.entity.config_entity import DataIngestionConfig


class Pipeline:
    def __init__(self,config:HousingConfig=HousingConfig())->None:
        try:
            self.config=config
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e

    def start_data_injection(self)->DataInjectionArtifacts:
        try:
            logging.info("Starting data injection pipeline")
            print(self.config.data_validation_config)
            data_injection_config=self.config.data_injection_config
            data_injeciotn=DataInjection(data_injection_config=data_injection_config)
            data_injection_artifacts=data_injeciotn.initiate_data_injection()
            logging.info(f'data injection artifact is [{data_injection_artifacts}]')
            return data_injection_artifacts
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e

    def start_data_validation(self,data_injection_artifacts:DataInjectionArtifacts)->bool:
        try:
            logging.info("Starting data validation pipeline")
            data_validation_config=self.config.data_validation_config
            data_validation=DataValidation(config=data_validation_config,data_injection_artifacts=data_injection_artifacts)
            run_or_not=data_validation.initiate_data_validation()
            logging.info(f'data validation output is [{run_or_not}]')
            return run_or_not
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e

    # def start_data_transformations(self)->DataTransformationArifacts:
    #     try:
    #         pass
    #     except Exception as e:
    #         raise CustomException(error_msg=e, error_details=sys) from  e
    
    # def start_model_training(self)->ModelTrainingArtifacts:
    #     try:
    #         pass
    #     except Exception as e:
    #         raise CustomException(error_msg=e, error_details=sys) from  e 

    # def start_model_evaluation(self)->ModelEvaluationArtifacts:
    #     try:
    #         pass
    #     except Exception as e:
    #         raise CustomException(error_msg=e, error_details=sys) from  e
    # def start_model_pusher(self)->ModelPusherArtifacts:
    #     try:
    #         pass
    #     except Exception as e:
    #         raise CustomException(error_msg=e, error_details=sys) from  e
    def run_pipeline(self):
        try:
            data_injection_artifacts=self.start_data_injection()
            run_or_not=self.start_data_validation(data_injection_artifacts=data_injection_artifacts)
            print(f'{run_or_not}')
            if run_or_not:
                pass
            else:
                logging.info(f'data set not match to data schema please check data set format')
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e


