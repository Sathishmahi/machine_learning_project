import os,sys
import json
import pickle
from housing.entity.artifacts_entity import ModelEvaluationArtifacts,ModelTrainingArtifacts
from housing.config.configuration import HousingConfig
from housing.logger import logging
from housing.exception import CustomException




class ModelEvaluation:
    def __init__(self,model_training_artifacts:ModelTrainingArtifacts,
                config:HousingConfig=HousingConfig(),)->None:

        self.model_evaluation_config=config.model_evalation_config
        self.model_training_artifacts=model_training_artifacts
        

    def to_read_json(self,json_info_file_path:str):
        try:
            if os.path.exists(json_info_file_path):
                with open(json_info_file_path,'r') as json_file:
                    json_content=json_content=json.load(json_file)
                    if len(json_content)==0:
                        raise Exception("file is empty")
                    
                    return json_content
                    
            raise FileNotFoundError('json file not found')
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)


    def initiate_model_evaluation(self,model_dir:str,json_info_file_path:str,base_score=0.10)->ModelEvaluationArtifacts:
        try:

            models_tuple=()
            to_change_or_not=False
            base_model_dir_list=model_dir.split('/')[:-2]
            base_model_dir='/'.join(base_model_dir_list)
            model_dir_items=[item for item in os.listdir(base_model_dir) if '.' not in item]
            print(f'model dir items {model_dir_items}')
            current_model_report_json_path=self.model_evaluation_config.current_model_report_json_path
            if len(model_dir_items)==0:
                raise Exception('models dir are empty')

            if len(model_dir_items)==1 and (not os.path.exists(current_model_report_json_path)):
                base_evaluation_dir_name=os.path.dirname(current_model_report_json_path)
                os.makedirs(base_evaluation_dir_name,exist_ok=True)
                with open(current_model_report_json_path,'w') as json_file:
                        model_info_over_all_dic=self.to_read_json(json_info_file_path)
                        json.dump(model_info_over_all_dic, json_file)
                return ModelEvaluationArtifacts(to_change_or_not=to_change_or_not, models_tuple=models_tuple)
            
            if len(model_dir_items)>=2:
                current_trained_models_info=self.to_read_json(json_info_file_path=json_info_file_path)
                current_model_info=self.to_read_json(current_model_report_json_path)
                current_model_info_keys=current_model_info.keys()
                no_of_cluster=len(os.listdir(self.model_training_artifacts.saved_model_dir_path))
                all_current_trained_trained_model_info_keys=list(current_trained_models_info.keys())[-no_of_cluster:]
                
                all_previous_model_path,all_replace_model_path=[],[]
                if len(current_model_info_keys)==len(all_current_trained_trained_model_info_keys):
                    for key_current_trained,key_current in zip(all_current_trained_trained_model_info_keys,current_model_info_keys):
                        score_current_trained=current_trained_models_info.get(key_current_trained)[-1]
                        score_current=current_model_info.get(key_current)[-1]
                        if (score_current_trained-score_current)>=base_score:
                            all_previous_model_path.append(key_current)
                            all_replace_model_path.append(key_current_trained)
                    if all_previous_model_path:
                        to_change_or_not=True
                        models_tuple=(all_previous_model_path,all_replace_model_path)
                        return ModelEvaluationArtifacts(to_change_or_not=to_change_or_not, models_tuple=models_tuple)
                    
                    return ModelEvaluationArtifacts(to_change_or_not=to_change_or_not, models_tuple=models_tuple)
                raise Exception("no of cluster not match")


        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    
    # def to_replace_model(self,model_dir:str,json_info_file_path:str,base_score=0.10)