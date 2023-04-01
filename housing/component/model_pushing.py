import os,sys
import json
import pickle
import shutil
from housing.entity.artifacts_entity import ModelEvaluationArtifacts,ModelPushinArtifacts
from housing.config.configuration import HousingConfig
from housing.logger import logging
from housing.exception import CustomException


class ModelPushing:
    def __init__(self,model_evaluation_artifacts:ModelEvaluationArtifacts,
                config:HousingConfig=HousingConfig())->None:

        
        self.model_pushing_config=config.model_pusher_config
        self.model_evalation_artifacts=model_evaluation_artifacts
        self.to_change_or_not=model_evaluation_artifacts.to_change_or_not
        self.models_tuple=model_evaluation_artifacts.models_tuple

    def initiate_model_pushing(self,cluster_file_path:str,len_of_model_training_dir:int,train_models_path=None,)->ModelPushinArtifacts:
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
                for model_name in os.listdir(train_models_path):
                    src_path=os.path.join(train_models_path,model_name)
                    src_copy,dst_copy=src_path,os.path.join(all_models_dir_path,model_name)
                    shutil.copyfile(src_copy, dst_copy)
                return model_pushing_artifacts
            if len_of_model_training_dir>=2:
                
                if not self.to_change_or_not:
                    return ModelPushinArtifacts(all_models_dir_path=all_models_dir_path,cluster_model_dir_path=cluster_model_dir)

                for old_file,new_file in zip(self.models_tuple[0],self.models_tuple[1]):
                    print(f'old file path ===== {old_file}')
                    print(f'new file path ===== {new_file}')
                    new_file_name=os.path.basename(new_file)
                    src_move,dst_move=old_file,old_production_model_dir 
                    shutil.move(src_move, dst_move)
                    src_copy,dst_copy=new_file,os.path.join(all_models_dir_path,new_file_name)
                    shutil.copyfile(src_copy, dst_copy)

                
                return model_pushing_artifacts
        
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)