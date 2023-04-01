import yaml
import os, sys
from housing.exception import CustomException
from housing.logger import logging
from housing.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    ModelPusherConfig,
    TrainingPipelineConfig,
)
from housing.utils.util import read_yaml
from housing.constant import *


class HousingConfig:
    def __init__(
        self,
        config_file_path: str = CONFIG_FILE_PATH,
        time_stamp: str = CURRENT_TIME_STAMP,
    ) -> None:
        try:
            self.time_stamp = time_stamp
            self.config_info = read_yaml(file_path=config_file_path)
            self.model_pipeline_config = self.get_data_model_pipeline_config()
            self.data_injection_config = self.get_data_injection_config()
            self.data_validation_config = self.get_data_validation_config()
            self.model_transformation_config = self.get_data_transformation_config()
            self.model_training_config = self.get_data_model_training_config()
            self.model_evalation_config = self.get_data_model_evaluation_config()
            self.model_pusher_config = self.get_data_model_push_config()
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from e

    def get_data_injection_config(self) -> DataIngestionConfig:
        try:
            print(f"------- {self.model_pipeline_config.artifact_dir}")
            data_injection_config_dic = self.config_info.get(DATA_INJECTION_CONFIG_KEY)
            ROOT_DIR_DATA_INJECTION = os.path.join(
                self.model_pipeline_config.artifact_dir,
                DATA_INJECTED_ARTIFACT_DIR,
                self.time_stamp,
            )
            downloaded_url = data_injection_config_dic.get(
                DATA_INJECTION_DATASET_DOWNLOAD_URL_KEY
            )

            raw_data_dir = os.path.join(
                ROOT_DIR_DATA_INJECTION,
                data_injection_config_dic.get(DATA_INJECTION_RAW_DATA_DIR_KEY),
            )

            tgz_download_dir = os.path.join(
                ROOT_DIR_DATA_INJECTION,
                data_injection_config_dic.get(DATA_INJECTION_TGZ_DATA_DIR_KEY),
            )

            injected_data_dir = os.path.join(
                ROOT_DIR_DATA_INJECTION,
                data_injection_config_dic.get(DATA_INJECTION_INJECTED_DATA_DIR_KEY),
            )

            train_data_dir = os.path.join(
                ROOT_DIR_DATA_INJECTION,
                data_injection_config_dic.get(DATA_INJECTION_INJECTED_DATA_DIR_KEY),
                data_injection_config_dic.get(DATA_INJECTION_TRAIN_DATA_DIR_KEY),
            )

            test_data_dir = os.path.join(
                ROOT_DIR_DATA_INJECTION,
                data_injection_config_dic.get(DATA_INJECTION_INJECTED_DATA_DIR_KEY),
                data_injection_config_dic.get(DATA_INJECTION_TEST_DATA_DIR_KEY),
            )

            data_injection_config = DataIngestionConfig(
                dataset_download_url=downloaded_url,
                tgz_download_dir=tgz_download_dir,
                raw_data_dir=raw_data_dir,
                ingested_train_dir=train_data_dir,
                ingested_test_dir=test_data_dir,
            )
            logging.info(f"data_injection_config : {data_injection_config}")

            return data_injection_config
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_config_dic = self.config_info.get(
                DATA_VALIDATION_CONFIG_KEY
            )
            DATA_VALIDATION_ROOT_DIR = os.path.join(
                self.model_pipeline_config.artifact_dir,
                data_validation_config_dic.get(DATA_VALIDATION_DIR_KEY),
                self.time_stamp,
            )
            schema_file_path = os.path.join(
                ROOT_DIR,
                data_validation_config_dic.get(DATA_VALIDATION_SCHEMA_DIR_KEY),
                data_validation_config_dic.get(DATA_VALIDATION_SCHEMA_FILE_NAME_KEY),
            )

            report_file_path = os.path.join(
                DATA_VALIDATION_ROOT_DIR,
                data_validation_config_dic.get(DATA_VALIDATION_REPORT_FILE_NAME_KEY),
            )

            report_page_file_path = os.path.join(
                DATA_VALIDATION_ROOT_DIR,
                data_validation_config_dic.get(
                    DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY
                ),
            )
            data_validation_config = DataValidationConfig(
                schema_file_path, report_file_path, report_page_file_path
            )
            logging.info(f"DataValidationConfig : {data_validation_config}")
            return data_validation_config
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transfomration_config_dic = self.config_info.get(
                DATA_TRANSFORMATION_CONFIG_KEY
            )
            DATA_TRANSFORMATION_ROOT_DIR = os.path.join(
                self.model_pipeline_config.artifact_dir,
                data_transfomration_config_dic.get(DATA_TRANSFORMATION_DIR_KEY),
                self.time_stamp,
            )
            add_bedroom_per_room = data_transfomration_config_dic.get(
                DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY
            )
            transformed_train_dir = os.path.join(
                DATA_TRANSFORMATION_ROOT_DIR,
                data_transfomration_config_dic.get(
                    DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY
                ),
                data_transfomration_config_dic.get(
                    DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY
                ),
            )
            transformed_test_dir = os.path.join(
                DATA_TRANSFORMATION_ROOT_DIR,
                data_transfomration_config_dic.get(
                    DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY
                ),
                data_transfomration_config_dic.get(
                    DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY
                ),
            )

            preprocessed_object_file_path = os.path.join(
                DATA_TRANSFORMATION_ROOT_DIR,
                data_transfomration_config_dic.get(
                    DATA_TRANSFORMATION_TRANSFORMED_PROCESSING_DIR_KEY
                ),
                data_transfomration_config_dic.get(
                    DATA_TRANSFORMATION_TRANSFORMED_PROCESSING_OBJECT_FILE_NAME_KEY
                ),
            )

            json_info_file_path = os.path.join(
                DATA_TRANSFORMATION_ROOT_DIR,
                data_transfomration_config_dic.get(DATA_TRANSFORMATION_JSON_TRAIN_PATH),
            )

            data_transformation_config = DataTransformationConfig(
                add_bedroom_per_room,
                transformed_train_dir,
                transformed_test_dir,
                preprocessed_object_file_path,
                json_info_file_path
            )

            logging.info(f"data_transformation_config : data_transformation_config")
            return data_transformation_config
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def get_data_model_training_config(self) -> ModelTrainerConfig:
        try:
            model_training_config_dic = self.config_info.get(MODEL_TRAINING_CONFIG_KEY)
            MODEL_TRAINING_ROOT_DIR = os.path.join(
                self.model_pipeline_config.artifact_dir,
                model_training_config_dic.get(MODEL_TRAINING_DIR_KEY),
                self.time_stamp,
            )

            trained_model_dir = os.path.join(
                MODEL_TRAINING_ROOT_DIR,
                model_training_config_dic.get(MODEL_TRAINING_TRAINED_MODEL_DIR_KEY),
                # model_training_config_dic.get(
                #     MODEL_TRAINING_TRAINED_MODEL_FILE_NAME_KEY
                # ),
            )
            base_accuracy = model_training_config_dic.get(
                MODEL_TRAINING_MODEL_BASE_ACCURACY_KEY
            )
            model_config_file_path = os.path.join(
                ROOT_DIR,
                model_training_config_dic.get(MODEL_TRAINING_MODEL_CONFIG_DIR_KEY),
                model_training_config_dic.get(
                    MODEL_TRAINING_MODEL_CONFIG_FILE_NAME_KEY
                ),
            )

            cluster_model_file_path=os.path.join(
                MODEL_TRAINING_ROOT_DIR,
                model_training_config_dic.get(MODEL_TRAINING_CLUSTER_DIR_KEY),
                model_training_config_dic.get(MODEL_TRAINING_CLUSTER_FILE_NAME_KEY),
            )

            model_info_json_file_path=os.path.join(
                MODEL_TRAINING_ROOT_DIR,
                model_training_config_dic.get(MODEL_TRAINING_MODEL_INFO_JSON_FILE_NAME_KEY)
            )
            model_training_config = ModelTrainerConfig(
                trained_model_dir, base_accuracy, model_config_file_path,
                model_info_json_file_path,
                cluster_model_file_path
            )
            logging.info(f"model_training_config : {model_training_config}")
            return model_training_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_evaluaton_config_dic = self.config_info.get(
                MODEL_EVALUATION_CONFIG_KEY
            )
            MODEL_EVALUATION_ROOT_DIR = os.path.join(
                self.model_pipeline_config.artifact_dir,
                model_evaluaton_config_dic.get(MODEL_EVALUATION_DIR_KEY),
                self.time_stamp,
            )
            model_evaluation_file_path = os.path.join(
                MODEL_EVALUATION_ROOT_DIR,
                model_evaluaton_config_dic.get(MODEL_EVALUATION_FILE_NAME_KEY),
            )
            model_evaluation_config = ModelEvaluationConfig(
                model_evaluation_file_path, self.time_stamp
            )
            logging.info(f"model_evaluation_config : {model_evaluation_config}")
            return model_evaluation_config

        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def get_data_model_push_config(self) -> ModelPusherConfig:
        try:
            model_pushing_config_dic = self.config_info.get(MODEL_PUSHER_CONFIG_KEY)
            model_export_dir_name = model_pushing_config_dic.get(
                MODEL_PUSHER_MODEL_EXPORT_DIR_KEY
            )
            MODEL_PUSHER_ROOT_DIR = os.path.join(
                self.model_pipeline_config.artifact_dir,
                model_pushing_config_dic.get(MODEL_PUSHER_DIR_KEY),
                self.time_stamp,
            )
            model_export_dir = os.path.join(
                MODEL_PUSHER_ROOT_DIR, MODEL_PUSHER_MODEL_EXPORT_DIR_KEY
            )
            model_pusher_config = ModelPusherConfig(export_dir_path=model_export_dir)

            logging.info(msg=f"model_pusher_config : {model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from e

    def get_data_model_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info.get(
                TRAINING_PIPELINE_CONFIG_KEY
            )
            artifact_dir = os.path.join(
                ROOT_DIR,
                training_pipeline_config.get(
                    TRAINING_PIPELINE_CONFIG_PIPELINE_NAME_KEY
                ),
                training_pipeline_config.get(TRAINING_PIPELINE_CONFIG_ARTIFACT_KEY),
            )
            training_pipe_line_config = TrainingPipelineConfig(
                artifact_dir=artifact_dir
            )
            logging.info(f"training_pipe_line_config : {training_pipe_line_config}")
            return training_pipe_line_config
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from e
