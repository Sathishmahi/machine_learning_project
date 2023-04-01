import os
import sys
from housing.config.configuration import HousingConfig
from housing.component.data_injection import DataInjection
from housing.component.data_validation import DataValidation
from housing.component.feature_engineering import DataTransformation
from housing.component.model_training import ModelTraining
from housing.logger import logging
from housing.component.model_evaluation import ModelEvaluation
from housing.component.model_pushing import ModelPushing
from housing.exception import CustomException
from housing.entity.artifacts_entity import (
    DataInjectionArtifacts,
    FeatureEngineeringArtifacts,
    DataValidationArtifacts,
    ModelTrainingArtifacts,
    ModelEvaluationArtifacts,
    ModelPushinArtifacts
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
            logging.info(f'finish model training and model training output is [{model_training_artifacts}]')
            return model_training_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from  e

    def start_model_evaluation(self,model_training_artifacts:ModelTrainingArtifacts)->ModelEvaluationArtifacts:
        try:
            logging.info(f'start model_evaluation')
            evaluation=ModelEvaluation(model_training_artifacts)
            model_dir=model_training_artifacts.saved_model_dir_path
            json_info_file_path=model_training_artifacts.ovel_all_model_training_json_file_path
            model_evalation_artifacts=evaluation.initiate_model_evaluation(model_dir, json_info_file_path)
            logging.info(f'finish model evalation and model training output is [{model_evalation_artifacts}]')
            return model_evalation_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from  e
    def start_model_pusher(self,model_evalation_artifacts:ModelEvaluationArtifacts,cluster_file_path:str,len_of_model_training_dir:int,
                        train_models_path:str)->ModelPushinArtifacts:
        try:
            model_pushing=ModelPushing(model_evaluation_artifacts=model_evalation_artifacts)
            model_pushing.initiate_model_pushing(cluster_file_path=cluster_file_path,len_of_model_training_dir=len_of_model_training_dir,
                                                train_models_path=train_models_path)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from  e


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

            model_evalation_artifacts=self.start_model_evaluation(model_training_artifacts=model_training_artifacts)
            print(f"{'='*10}    finish model evaluation       {'='*10}")
            model_dir=model_training_artifacts.saved_model_dir_path
            base_model_dir_list=model_dir.split('/')[:-2]
            base_model_dir='/'.join(base_model_dir_list)
            model_dir_items=[item for item in os.listdir(base_model_dir) if '.' not in item]
            len_of_model_training_dir,cluster_file_path=len(model_dir_items),self.config.model_training_config.cluster_model_file_path

            model_pushing_artifacts=self.start_model_pusher(model_evalation_artifacts=model_evalation_artifacts,cluster_file_path=cluster_file_path, 
                                                            len_of_model_training_dir=len_of_model_training_dir,train_models_path=model_dir)

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e
