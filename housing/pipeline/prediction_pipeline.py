from housing.exception import CustomException
from housing.logger import logging
from housing.entity.artifacts_entity import ModelPushinArtifacts
import os,sys
import pandas as pd
import numpy as np
from typing import List
import pickle


class ModelPrediction:
    def __init__(self,model_pushing_artifacts:ModelPushinArtifacts):
        try:
            self.model_pushing_artifacts=model_pushing_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    def to_cluster_model(self,predicted_data:pd.DataFrame,cluster_model_path:str)->np.array:
        try:
            predicted_data_copy=predicted_data.copy()
            if not os.path.exists(cluster_model_path):
                raise FileNotFoundError('cluster model not found')
            with open(cluster_model_path,'rb') as pickle_file:
                cluster_model=pickle.load(pickle_file)
            cluster_no=cluster_model.predict(predicted_data)
            predicted_data_copy['cluster_no']=cluster_no

            return predicted_data_copy

        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def to_predict(self,predicted_data_with_cluster_no:pd.DataFrame,models_dir_path:str)->pd.DataFrame:

        try:
            predicted_data_with_cluster_no_copy=predicted_data_with_cluster_no.copy()
            if not os.path.exists(models_dir_path):
                raise FileNotFoundError('model dir not found')
            predicted_data_with_cluster_no_copy['out_come']=0.0
            all_models=os.listdir(models_dir_path)
            for model in all_models:
                group=model.split('_')[1]
                temp_df=predicted_data_with_cluster_no_copy.iloc['cluster_no'==group].drop(columns=['cluster_no','out_come'])
                if len(temp_df)>=1:
                    with open(os.path.join(models_dir_path,model),'rb') as pickle_file:
                        loaded_model=pickle.load(pickle_file)
                    predicted_out=loaded_model.predict(temp_df)
                    predicted_data_with_cluster_no_copy.iloc['cluster_no'==group,'out_come']=predicted_out
            return predicted_data_with_cluster_no.drop('cluster_no')
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    
    def initiate_data_prediction(self,predicted_data:pd.DataFrame)->pd.DataFrame:
        try:
            models_dir_path=self.model_pushing_artifacts.all_models_dir_path
            cluster_model_path=self.model_pushing_artifacts.cluster_model_dir_path

            predicted_data_with_cluster_no=self.to_cluster_model(predicted_data=predicted_data, cluster_model_path=cluster_model_path)
            after_predict_data=self.to_predict(predicted_data_with_cluster_no=predicted_data_with_cluster_no, models_dir_path=models_dir_path)
            return after_predict_data
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)










            

            
