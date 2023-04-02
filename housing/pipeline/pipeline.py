import os
import sys
import json
import pandas as pd
from housing.entity.config_entity import DataTransformationConfig
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
from housing.pipeline.prediction_pipeline import ModelPrediction
from housing.constant import PREDICTION_HELPER_JSON_FILE_NAME


class Pipeline:
    def __init__(self, config: HousingConfig = HousingConfig(),is_predicton:bool=True) -> None:
        try:
            self.config = config
            self.is_predicton=is_predicton
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e


    def to_write_json(self,json_file_path:str,content:dict):
        try:
            already_present_dic=dict()
            if os.path.exists(json_file_path):
                with open(json_file_path,'r') as json_file:
                    already_present_dic=json.load(json_file)
                with open(json_file_path,'w') as json_file:
                    already_present_dic.update(content)
                    json.dump(already_present_dic,json_file)

            with open(json_file_path,'w') as json_file:
                    json.dump(already_present_dic,json_file)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
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
        latest_training_data_transformation_info_json_path:str=None,
        saved_model_dir:str=None,
        save_prediction_data_dir=None,
        predicted_data:pd.DataFrame=None,
        is_prediction_data=False
    ) -> FeatureEngineeringArtifacts:
        try:
            logging.info(f"start feature engineering")
            data_transformation = DataTransformation(
                data_validation_artifacts=data_validation_artifacts,
                data_injection_artifacts=data_injection_artifacts,
            )

            feature_engineering_artifacts = (
                data_transformation.initiate_data_transformation(is_prediction_data=is_prediction_data,prediction_df=predicted_data,
                save_prediction_data_dir=save_prediction_data_dir,latest_training_data_transformation_info_json_path=latest_training_data_transformation_info_json_path)
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
                        train_models_path:str,model_info_over_all_json_file_path:str)->ModelPushinArtifacts:
        try:
            model_pushing=ModelPushing(model_evaluation_artifacts=model_evalation_artifacts)
            model_pushing.initiate_model_pushing(cluster_file_path=cluster_file_path,len_of_model_training_dir=len_of_model_training_dir,
                                                train_models_path=train_models_path,model_info_over_all_json_file_path=model_info_over_all_json_file_path)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from  e


    def run_pipeline(self):
        try:

            if not self.is_predicton:
                data_injection_artifacts = self.start_data_injection()
                print(f"{'='*10}    finish data injection   {'='*10}")
                self.to_write_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME, content={'data_injection_artifacts':list(data_injection_artifacts)})
                data_validation_artifacts = self.start_data_validation(
                    data_injection_artifacts=data_injection_artifacts
                )
                self.to_write_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME, content={'data_validation_artifacts':list(data_validation_artifacts)})

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
                self.to_write_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME, content={'feature_engineering_artifacts':list(feature_engineering_artifacts)})

                print(f"{'='*10}    finish feature engineering       {'='*10}")
                model_training_artifacts=self.start_model_training(feature_engineering_artifacts=feature_engineering_artifacts)
                print(f"{'='*10}    finish model training       {'='*10}")
                self.to_write_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME, content={'model_training_artifacts':list(model_training_artifacts)})
                
                model_evalation_artifacts=self.start_model_evaluation(model_training_artifacts=model_training_artifacts)
                print(f"{'='*10}    finish model evaluation       {'='*10}")
                self.to_write_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME, content={'model_evalation_artifacts':list(model_evalation_artifacts)})

                model_dir=model_training_artifacts.saved_model_dir_path
                base_model_dir_list=model_dir.split('/')[:-2]
                base_model_dir='/'.join(base_model_dir_list)
                model_dir_items=[item for item in os.listdir(base_model_dir) if '.' not in item]
                len_of_model_training_dir,cluster_file_path=len(model_dir_items),self.config.model_training_config.cluster_model_file_path

                model_info_over_all_json_file_path=model_training_artifacts.ovel_all_model_training_json_file_path
                model_pushing_artifacts=self.start_model_pusher(model_evalation_artifacts=model_evalation_artifacts,cluster_file_path=cluster_file_path, 
                                                                len_of_model_training_dir=len_of_model_training_dir,train_models_path=model_dir,
                                                                model_info_over_all_json_file_path=model_info_over_all_json_file_path)
                self.to_write_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME, content={'model_pushing_artifacts':list(model_pushing_artifacts)})

                print(f"{'='*10}    finish model pushing       {'='*10}")
            else:
                df=pd.read_csv('/config/workspace/housing/artifact/data_injection/2023_04_02_11_12_43/ingested_data/test/housing.csv',nrows=1000)
                if not os.path.exists(PREDICTION_HELPER_JSON_FILE_NAME):
                    raise FileNotFoundError('helper json file not found')
                with open(PREDICTION_HELPER_JSON_FILE_NAME,'r') as json_file:
                    feature_engineering_artifacts_list=json.load(json_file).get('feature_engineering_artifacts')
                model_dir,trained_model_info_json_path=feature_engineering_artifacts_list[2],feature_engineering_artifacts_list[-1]
                
                li=self.start_data_transformations(data_injection_artifacts=None, saved_model_dir=model_dir,
                data_validation_artifacts=None,predicted_data=df,is_prediction_data=self.is_predicton,
                latest_training_data_transformation_info_json_path=trained_model_info_json_path)
                # model_prediction=ModelPrediction(model_pushing_artifacts)
                # self.start_data_transformations(data_injection_artifacts, data_validation_artifacts)
                # model_prediction.initiate_data_prediction(predicted_data=pd.read_csv())
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e
