import os
from datetime import datetime


ROOT_DIR = os.getcwd()
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)
print(CONFIG_FILE_PATH)
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"


##training pipe line related constant variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_CONFIG_ARTIFACT_KEY = "artifact_dir"
TRAINING_PIPELINE_CONFIG_PIPELINE_NAME_KEY = "pipeline_name"


### data injection related constant variable
DATA_INJECTION_CONFIG_KEY = "data_ingestion_config"
DATA_INJECTED_ARTIFACT_DIR = "data_injection"
DATA_INJECTION_DATASET_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INJECTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INJECTION_TGZ_DATA_DIR_KEY = "tgz_download_dir"
DATA_INJECTION_INJECTED_DATA_DIR_KEY = "ingested_dir"
DATA_INJECTION_TRAIN_DATA_DIR_KEY = "ingested_train_dir"
DATA_INJECTION_TEST_DATA_DIR_KEY = "ingested_test_dir"


###model pushing related constant variable
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_DIR_KEY = "model_pusher_dir"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"


###data validation costant variable

DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_DIR_KEY = "data_validation_dir"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"


### data tranformation related constant variable
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR_KEY = "data_transformation_dir"
DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY = "add_bedroom_per_room"
DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_TRANSFORMED_PROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_TRANSFORMED_PROCESSING_OBJECT_FILE_NAME_KEY = (
    "preprocessed_object_file_name"
)
DATA_TRANSFORMATION_JSON_TRAIN_PATH = "json_info_file_path"


## report json related var

ALL_NULL_VALUES_COLUMNS_KEY = "all_null_value_columns"
ALL_MULTICOLINEARITY_COLUMNS_DICT_KEY = "all_multi_colinearity_features_dict"
ALL_NEGATIVE_CORR_COLUMNS_KEY = "all_negative_corr_col_dic"
UNNORMAL_DIST_COLUMNS_KEY = "un_normal_dist_col"
ALL_DISCRETE_COLUMNS_KEY = "all_cat_columns"


## train transformation json file related var
AFTER_HANDLE_NEGATIVE_CORRELATION_TRAIN_DF_COLUMNS_LIST = (
    "after_handle_negative_correlation_train_df_columns"
)
ALL_UNNORMAL_DISTRIBUTION_COLUMNS_LIST = "all_non_normal_distribution_columns_list"
AFTER_HANDLE_THE_MULTICOLINEARITY_TRAIN_DF_COLUMNS_LIST = (
    "after_handle_multi_colineraity_train_df_columns"
)
HANDLE_CAT_FEATURES_DICT = "handle_cat_features_dict"
AFTER_REMOVE_ONE_STD_TRAIN_DIR_LIST = "after_handle_one_std_train_df_columns"


### model training config related constant variable
MODEL_TRAINING_DICT_KEY = "model_training"
MODEL_TRAINING_MODEL_NAMES_LIST = "model_name_list"


### model training config related constant variable
MODEL_TRAINING_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINING_DIR_KEY = "model_training_dir"
MODEL_TRAINING_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINING_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINING_MODEL_BASE_ACCURACY_KEY = 0.6
MODEL_TRAINING_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINING_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"


###model evaluation related constant variable
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_DIR_KEY = "model_evaluation_dir"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"


###  Validation schema related constant variable

COLUMNS_KEY = "columns"
NUMERICAL_COLUMNS_KEY = "numeric_columns"
CATEGORICAL_COLUMNS_KEY = "categorical_columns"
TARGET_COLUMN_KEY = "target_column"
DOMAIN_VALUE_KEY = "domain_value"
INSIDE_DOMAIN_VALUE_KEY = "ocean_proximity"

### all plot related information
HISTOGRAM_PLOT_DIR = os.path.join("histogram_plots", CURRENT_TIME_STAMP)
HISTOGRAM_PLOT_DIR = os.path.join("histogram", CURRENT_TIME_STAMP)
BAR_PLOT_DIR = os.path.join("barplots", CURRENT_TIME_STAMP)
BOX_PLOT_DIR = os.path.join("boxplots", CURRENT_TIME_STAMP)
CORR_PLOT_DIR = os.path.join("corr", CURRENT_TIME_STAMP)
DIST_PLOT_DIR = os.path.join("distplots", CURRENT_TIME_STAMP)
SCATTER_PLOT_DIR = os.path.join("scatterplots", CURRENT_TIME_STAMP)
COUNTS_PLOT_DIR = os.path.join("countplots", CURRENT_TIME_STAMP)
