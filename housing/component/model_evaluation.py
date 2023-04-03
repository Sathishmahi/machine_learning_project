import json
import os
import pickle
import sys

from housing.config.configuration import HousingConfig
from housing.entity.artifacts_entity import (ModelEvaluationArtifacts,
                                             ModelTrainingArtifacts)
from housing.exception import CustomException
from housing.logger import logging


class ModelEvaluation:

    def __init__(self,model_training_artifacts:ModelTrainingArtifacts,
                config:HousingConfig=HousingConfig(),)->None:
        """
        ModelEvaluation to evaluate or compare models current trained models and already production models
    

        Args:
            model_training_artifacts (ModelTrainingArtifacts): all model training model paths
            config (HousingConfig, optional): all config class. Defaults to HousingConfig().

        Raises:
            CustomException: 
        """
        try:
            self.model_evaluation_config=config.model_evalation_config
            self.model_training_artifacts=model_training_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
        

    def to_read_json(self,json_info_file_path:str)->dict:
        """
        to_read_json to read json file

        Args:
            json_info_file_path (str): path of json file
        Raises:
            Exception: 
            FileNotFoundError: 
            CustomException: 

        Returns:
            dict: content of json
        """
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


    def initiate_model_evaluation(self,model_dir:str,json_info_file_path:str,base_score:float=0.10)->ModelEvaluationArtifacts:
        
        """
        initiate_model_evaluation to compare current trained and production models 

        Attrs:
            model_dir(str) :old models dir
            json_info_file_path(str) :path for all models info contain json file
            base_score(float) :base score of model change or not 
            if current trained model score and current production model score more the base to /
            change the production model replce to current model.

        Note :replaced model move to old model dir 
        Raises:
            Exception: 
            CustomException: 

        Returns:
            ModelEvaluationArtifacts: to return all changeable models path(new[currnt trained model] and old[production model] model path)
        """
        
        try:  
            base_model_dir_list=model_dir.split('/')[:-2]
            base_model_dir='/'.join(base_model_dir_list)
            model_dir_items=[item for item in os.listdir(base_model_dir) if '.' not in item]
            print(f'model dir items {model_dir_items}')
            models_tuple=()
            to_change_or_not=False
            current_model_report_json_path=self.model_evaluation_config.current_model_report_json_path
            if len(model_dir_items)==0:
                raise Exception('models dir are empty')

            if len(model_dir_items)==1 and (not os.path.exists(current_model_report_json_path)):
                base_evaluation_dir_name=os.path.dirname(current_model_report_json_path)
                os.makedirs(base_evaluation_dir_name,exist_ok=True)
                with open(current_model_report_json_path,'w') as json_file:
                        model_info_over_all_dic=self.to_read_json(json_info_file_path)
                        json.dump(model_info_over_all_dic, json_file)

                return ModelEvaluationArtifacts(to_change_or_not=to_change_or_not, models_tuple=models_tuple,model_evaluation_current_model_info_file_path=current_model_report_json_path)
            
            if len(model_dir_items)>=2:
                current_trained_models_info=self.to_read_json(json_info_file_path=json_info_file_path)
                current_model_info=self.to_read_json(current_model_report_json_path)
                current_model_info_keys=current_model_info.keys()
                no_of_cluster=len(os.listdir(self.model_training_artifacts.saved_model_dir_path))
                all_current_trained_trained_model_info_keys=list(current_trained_models_info.keys())[-no_of_cluster:]
                
                all_previous_model_path,all_replace_model_path=[],[]
                if len(current_model_info_keys)==len(all_current_trained_trained_model_info_keys):
                    for key_current in current_model_info_keys:
                        group_curent=os.path.basename(key_current).split('_')[1]

                        for key_current_trained in all_current_trained_trained_model_info_keys:
                            group_curent_trained=os.path.basename(key_current_trained).split('_')[1]

                            if group_curent==group_curent_trained:
                                score_current_trained=current_trained_models_info.get(key_current_trained)[-1]
                                score_current=current_model_info.get(key_current)[-1]

                                if (score_current_trained-score_current)>=base_score:
                                    all_previous_model_path.append(key_current)
                                    all_replace_model_path.append(key_current_trained)

                    if all_previous_model_path:
                        to_change_or_not=True
                        models_tuple=(all_previous_model_path,all_replace_model_path)
                        return ModelEvaluationArtifacts(to_change_or_not=to_change_or_not, models_tuple=models_tuple,model_evaluation_current_model_info_file_path=current_model_report_json_path)
                    
                    return ModelEvaluationArtifacts(to_change_or_not=to_change_or_not, models_tuple=models_tuple,model_evaluation_current_model_info_file_path=current_model_report_json_path)
                raise Exception("no of cluster not match")


        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    