import os
import sys
from housing.config.configuration import HousingConfig
from housing.component.data_injection import DataInjection
from housing.component.data_validation import DataValidation
from housing.component.feature_engineering import DataTransformation
from housing.component.model_training import ModelTraining
from housing.logger import logging

from housing.exception import CustomException
from housing.entity.artifacts_entity import (
    DataInjectionArtifacts,
    FeatureEngineeringArtifacts,
    DataValidationArtifacts,
    ModelTrainingArtifacts
)
from housing.entity.config_entity import DataIngestionConfig


class Pipeline:
    def __init__(self, config: HousingConfig = HousingConfig()) -> None:
        try:
            self.config = config
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def start_data_injection(self) -> DataInjectionArtifacts:
        try:
            logging.info("Starting data injection pipeline")
            print(self.config.data_validation_config)
            data_injection_config = self.config.data_injection_config
            data_injeciotn = DataInjection(data_injection_config=data_injection_config)
            data_injection_artifacts = data_injeciotn.initiate_data_injection()
            logging.info(f"finish data injection and data injection artifact is [{data_injection_artifacts}]")
            return data_injection_artifacts
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def start_data_validation(
        self, data_injection_artifacts: DataInjectionArtifacts
    ) -> bool:
        try:
            logging.info("Starting data validation pipeline")
            data_validation_config = self.config.data_validation_config
            data_validation = DataValidation(
                config=data_validation_config,
                data_injection_artifacts=data_injection_artifacts,
            )
            data_validation_artifacts = data_validation.initiate_data_validation()
            logging.info(f"finish data validation and data validation output is [{data_validation_artifacts}]")
            return data_validation_artifacts
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def start_data_transformations(
        self,
        data_injection_artifacts: DataInjectionArtifacts,
        data_validation_artifacts: DataValidationArtifacts,
    ) -> FeatureEngineeringArtifacts:
        try:
            logging.info(f"start feature engineering")
            data_transformation = DataTransformation(
                data_validation_artifacts=data_validation_artifacts,
                data_injection_artifacts=data_injection_artifacts,
            )

            feature_engineering_artifacts = (
                data_transformation.initiate_data_transformation()
            )
            logging.info(
                f"finish eature engineering and  eature engineering output is [{feature_engineering_artifacts}]"
            )
            return feature_engineering_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from e

    def start_model_training(self,feature_engineering_artifacts:FeatureEngineeringArtifacts)->ModelTrainingArtifacts:
        try:
            logging.info(f'start model training')
            training=ModelTraining(feature_engineering_artifacts)
            model_training_artifacts=training.initiate_model_training()
            return model_training_artifacts
            logging.info(f'finish model training and model training output is [{model_training_artifacts}]')
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from  e

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
            data_injection_artifacts = self.start_data_injection()
            print(f"{'='*10}    finish data injection   {'='*10}")
            data_validation_artifacts = self.start_data_validation(
                data_injection_artifacts=data_injection_artifacts
            )
            if data_validation_artifacts.all_correct_or_not:
                pass
            else:
                logging.info(
                    f"data set not match to data schema please check data set format"
                )
                raise Exception(
                    f"data set not match to data schema please check data set format"
                )
            print(f"{'='*10}    finish data validaton       {'='*10}")
            feature_engineering_artifacts = self.start_data_transformations(
                data_injection_artifacts=data_injection_artifacts,
                data_validation_artifacts=data_validation_artifacts,
            )
            print(f"{'='*10}    finish feature engineering       {'='*10}")
            model_training_artifacts=self.start_model_training(feature_engineering_artifacts=feature_engineering_artifacts)
            print(f"{'='*10}    finish model training       {'='*10}")
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e
