import os
from datetime import datetime


ROOT_DIR=os.getcwd()
CONFIG_DIR="config"
CONFIG_FILE_NAME="config.yaml"
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)
print(CONFIG_FILE_PATH)
CURRENT_TIME_STAMP=f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"


##training pipe line related constant variable
TRAINING_PIPELINE_CONFIG_KEY="training_pipeline_config"
TRAINING_PIPELINE_CONFIG_ARTIFACT_KEY="artifact_dir"
TRAINING_PIPELINE_CONFIG_PIPELINE_NAME_KEY="pipeline_name"


### data injection related constant variable
DATA_INJECTION_CONFIG_KEY="data_ingestion_config"
DATA_INJECTED_ARTIFACT_DIR="data_injection"
DATA_INJECTION_DATASET_DOWNLOAD_URL_KEY="dataset_download_url"
DATA_INJECTION_RAW_DATA_DIR_KEY="raw_data_dir"
DATA_INJECTION_TGZ_DATA_DIR_KEY="tgz_download_dir"
DATA_INJECTION_INJECTED_DATA_DIR_KEY="ingested_dir"
DATA_INJECTION_TRAIN_DATA_DIR_KEY="ingested_train_dir"
DATA_INJECTION_TEST_DATA_DIR_KEY="ingested_test_dir"




###model pushing related constant variable
MODEL_PUSHER_CONFIG_KEY="model_pusher_config"
MODEL_PUSHER_DIR_KEY="model_pusher_dir"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY="model_export_dir"


###data validation costant variable

DATA_VALIDATION_CONFIG_KEY="data_validation_config"
DATA_VALIDATION_DIR_KEY="data_validation_dir"
DATA_VALIDATION_SCHEMA_DIR_KEY="schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY="schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY="report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY="report_page_file_name"


### data tranformation related constant variable
DATA_TRANSFORMATION_CONFIG_KEY="data_transformation_config"
DATA_TRANSFORMATION_DIR_KEY="data_transformation_dir"
DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY="add_bedroom_per_room"
DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY="transformed_dir"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY="transformed_train_dir"
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY="transformed_test_dir"
DATA_TRANSFORMATION_TRANSFORMED_PROCESSING_DIR_KEY="preprocessing_dir"
DATA_TRANSFORMATION_TRANSFORMED_PROCESSING_OBJECT_FILE_NAME_KEY="preprocessed_object_file_name"


### model training config related constant variable
MODEL_TRAINING_CONFIG_KEY="model_trainer_config"
MODEL_TRAINING_DIR_KEY="model_training_dir"
MODEL_TRAINING_TRAINED_MODEL_DIR_KEY="trained_model_dir"
MODEL_TRAINING_TRAINED_MODEL_FILE_NAME_KEY="model_file_name"
MODEL_TRAINING_MODEL_BASE_ACCURACY_KEY=0.6
MODEL_TRAINING_MODEL_CONFIG_DIR_KEY="model_config_dir"
MODEL_TRAINING_MODEL_CONFIG_FILE_NAME_KEY="model_config_file_name"


###model evaluation related constant variable
MODEL_EVALUATION_CONFIG_KEY="model_evaluation_config"
MODEL_EVALUATION_DIR_KEY="model_evaluation_dir"
MODEL_EVALUATION_FILE_NAME_KEY="model_evaluation_file_name"




