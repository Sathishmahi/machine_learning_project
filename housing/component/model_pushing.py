import json
import os
import pickle
import shutil
import sys

from housing.config.configuration import HousingConfig
from housing.entity.artifacts_entity import (ModelEvaluationArtifacts,
                                             ModelPushinArtifacts)
from housing.exception import CustomException
from housing.logger import logging


class ModelPushing:
    def __init__(self,model_evaluation_artifacts:ModelEvaluationArtifacts,
                config:HousingConfig=HousingConfig())->None:
        """
        ModelPushing to push the models into production

        Args:
            model_evaluation_artifacts (ModelEvaluationArtifacts): model_evaluation_artifacts  
            config (HousingConfig, optional): all config class. Defaults to HousingConfig().
        """
        
        self.model_pushing_config=config.model_pusher_config
        self.model_evalation_artifacts=model_evaluation_artifacts
        self.to_change_or_not=model_evaluation_artifacts.to_change_or_not
        self.models_tuple=model_evaluation_artifacts.models_tuple

    def initiate_model_pushing(self,cluster_file_path:str,len_of_model_training_dir:int,
                                model_info_over_all_json_file_path:str,train_models_path=None,)->ModelPushinArtifacts:
        """
       initiate_model_pushing  to change the all less perform models

        Args:
            cluster_file_path (str): cluster file path
            len_of_model_training_dir (int): len of model training dir whether len == 1  first time training was /
            happen so all models directly move to production

            model_info_over_all_json_file_path (str): json file path for all models info
            train_models_path (_type_, optional): all currently trained models path. Defaults to None.

        Raises:
            CustomException: 

        Returns:
            ModelPushinArtifacts: to return path for cluster and production models 
        """
        current_model_report_json_path=self.model_evalation_artifacts.model_evaluation_current_model_info_file_path
        try:

            all_models_dir_path=self.model_pushing_config.production_models_dir
            cluster_model_dir=self.model_pushing_config.cluster_model_dir
            old_production_model_dir=self.model_pushing_config.old_production_model_dir
            model_pushing_artifacts=ModelPushinArtifacts(all_models_dir_path=all_models_dir_path, 
                                                            cluster_model_dir_path=cluster_model_dir)
            all_dir_path=[all_models_dir_path,cluster_model_dir,old_production_model_dir]
            for path in all_dir_path:
                os.makedirs(path,exist_ok=True)
            cluster_file_name=os.path.basename(cluster_file_path)
            src_copy,dst_copy=cluster_file_path,os.path.join(cluster_model_dir,cluster_file_name)
            shutil.copyfile(src_copy, dst_copy)

            if len_of_model_training_dir==1:
                print(f'inside the len = 1 ')
                keys,alternative_keys=[],[]
                for model_name in os.listdir(train_models_path):
                    src_path=os.path.join(train_models_path,model_name)
                    
                    src_copy,dst_copy=src_path,os.path.join(all_models_dir_path,model_name)
                    keys.append(src_path)
                    alternative_keys.append(dst_copy)
                    shutil.copyfile(src_copy, dst_copy)

                dic=dict()
                with open(current_model_report_json_path,'r') as json_file:
                    dic=json.load(json_file)
                with open(current_model_report_json_path,'w') as json_file:
                    print(f'new evaluation dic {alternative_keys}     {dic.values()}')
                    new_dic=dict(zip(alternative_keys,dic.values()))
                    json.dump(new_dic,json_file)
                return model_pushing_artifacts
                
            if len_of_model_training_dir>=2:
                
                if not self.to_change_or_not:
                    return ModelPushinArtifacts(all_models_dir_path=all_models_dir_path,cluster_model_dir_path=cluster_model_dir)
                with open(model_info_over_all_json_file_path,'r') as json_file:
                    json_content=json.load(json_file)
                keys,alternative_keys,score_get_key=[],[],[]
                for old_file,new_file in zip(self.models_tuple[0],self.models_tuple[1]):
                    print(f'old file path ===== {old_file}')
                    print(f'new file path ===== {new_file}')
                    new_file_name=os.path.basename(new_file)
                    old_file_name=os.path.basename(old_file)
                    
                    src_move,dst_move=old_file,old_production_model_dir 
                    shutil.move(src_move, dst_move)
                    src_copy,dst_copy=new_file,os.path.join(all_models_dir_path,new_file_name)
                    keys.append(old_file)
                    alternative_keys.append(dst_copy)
                    score_get_key.append(new_file)
                    shutil.copyfile(src_copy, dst_copy)
                dic=dict()
                with open(current_model_report_json_path,'r') as json_file:
                    dic=json.load(json_file)
                with open(current_model_report_json_path,'w') as json_file:
                    
                    for key,alter_key,get_key in zip(keys,alternative_keys,score_get_key):
                        dic.pop(key)
                        dic.update({alter_key:json_content.get(get_key)})
                    json.dump(dic,json_file)
                
                return model_pushing_artifacts
        
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)