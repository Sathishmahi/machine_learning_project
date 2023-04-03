import os
import pickle
import sys
from typing import List

import numpy as np
import pandas as pd

from housing.entity.artifacts_entity import ModelPushinArtifacts
from housing.exception import CustomException
from housing.logger import logging


class ModelPrediction:
    def __init__(self,model_pushing_artifacts_list:list):
        """
        ModelPrediction to predict data 
        Args:
            model_pushing_artifacts_list (list): model pushing artifacts list

        Raises:
            CustomException: 
        """
        try:
            self.model_pushing_artifacts_list=model_pushing_artifacts_list
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    def to_cluster_model(self,predicted_data:pd.DataFrame,cluster_model_path:str)->pd.DataFrame:
        """
        to_cluster_model to cluster the predicted data
        
        Args:
            predicted_data (pd.DataFrame): predicted data
            cluster_model_path (str): cluster model path

        Raises:
            FileNotFoundError: 
            CustomException: 

        Returns:
            pd.DataFrame: to return predicted data with cluster no
        """
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
        """
        to_predict to predict the data 

        Args:
            predicted_data_with_cluster_no (pd.DataFrame): predited data with cluster no
            models_dir_path (str): all trained models dir

        Raises:
            FileNotFoundError: 
            CustomException: 

        Returns:
            pd.DataFrame: 
        """
        try:
            predicted_data_with_cluster_no_copy=predicted_data_with_cluster_no.copy()
            if not os.path.exists(models_dir_path):
                raise FileNotFoundError('model dir not found')
            predicted_data_with_cluster_no_copy['out_come']=0.0
            all_models=os.listdir(models_dir_path)
            for model in all_models:
                group=model.split('_')[1]
                print(f'group  {group}')
                temp_df=predicted_data_with_cluster_no_copy[predicted_data_with_cluster_no_copy['cluster_no']==int(group)].drop(columns=['cluster_no','out_come'])
                print(temp_df.head(2))
                if len(temp_df)>=1:
                    with open(os.path.join(models_dir_path,model),'rb') as pickle_file:
                        loaded_model=pickle.load(pickle_file)
                    predicted_out=loaded_model.predict(temp_df)                    
                    predicted_data_with_cluster_no_copy.loc[predicted_data_with_cluster_no_copy['cluster_no']==int(group),'out_come']=predicted_out
            return predicted_data_with_cluster_no_copy.drop(columns='cluster_no')
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    
    def initiate_data_prediction(self,predicted_data:pd.DataFrame)->pd.DataFrame:
        """
        initiate_data_prediction to combine all prediction function

        Args:
            predicted_data (pd.DataFrame): predicted data

        Raises:
            CustomException: 

        Returns:
            pd.DataFrame: predited data with out-come
        """
        try:
            models_dir_path=self.model_pushing_artifacts_list[0]
            cluster_dir_path=self.model_pushing_artifacts_list[1]
            cluster_model_path=os.path.join(cluster_dir_path,os.listdir(cluster_dir_path)[0])
            predicted_data_with_cluster_no=self.to_cluster_model(predicted_data=predicted_data, cluster_model_path=cluster_model_path)
            after_predict_data=self.to_predict(predicted_data_with_cluster_no=predicted_data_with_cluster_no, models_dir_path=models_dir_path)
            return after_predict_data
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)










            

            
