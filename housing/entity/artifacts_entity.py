from collections import namedtuple

DataInjectionArtifacts = namedtuple(
    "DataInjectionArtifacts",
    ["train_file_path", "test_file_path", "is_injected", "message"],
)

DataValidationArtifacts = namedtuple(
    "DataValidationArtifacts", ["all_correct_or_not", "json_report_file_path"]
)

FeatureEngineeringArtifacts = namedtuple(
    "FeatureEngineeringArtifacts",
    [
        "transformed_train_file_path",
        "transformed_test_file_path",
        "tranformation_model_path",
        "is_done_for_FE",
        "message",
        "trained_model_info_json_path"
    ]
)

ModelTrainingArtifacts = namedtuple("DataTrainingArtifacts", [
    "ovel_all_model_training_json_file_path",
    "saved_model_dir_path"
])

ModelEvaluationArtifacts=namedtuple("ModelEvaluationArtifacts", [
    "to_change_or_not",
    "models_tuple",
    "model_evaluation_current_model_info_file_path"
])

ModelPushinArtifacts=namedtuple("ModelPushinArtifacts", [
    'all_models_dir_path',"cluster_model_dir_path"
])