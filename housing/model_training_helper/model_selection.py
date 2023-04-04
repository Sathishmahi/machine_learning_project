import json
import os
import pickle
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV

from housing.constant import *
from housing.constant.experiment_models import all_models_dict
from housing.constant.hyper_parameters import all_params_dict
from housing.entity.artifacts_entity import ModelTrainingArtifacts
from housing.exception import CustomException
from housing.logger import logging


class ReturnParams:

    def __init__(self,all_params_dict:dict=all_params_dict,
                all_models_dict:dict=all_models_dict)->None:
        """
        ReturnParams to return best possible parameter based on the model performance by using RandomSearchCV

        Args:
            all_params_dict (dict, optional): all hyper parameter dict. Defaults to all_params_dict.
            all_models_dict (dict, optional): all models dict. Defaults to all_models_dict.

        Raises:
            CustomException: 
        """
        try:
            self.all_params_dict=all_params_dict
            self.all_models_dict=all_models_dict
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    def read_json(self,json_file_path:str)->dict:
        """
        read_json to read a json file

        Args:
            json_file_path (str): path of the json file

        Raises:
            CustomException: 

        Returns:
            dict: to return json content
        """
        try:
            with open(json_file_path,'r') as json_file:
                json_content=json.load(json_file)
            return json_content
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    def _return_params(self,model_name:str,json_file_path:str=None)->dict:
        """
        _return_params to return all params 

        Args:
            model_name (str): model name 
            json_file_path (str, optional): . Defaults to None.

        Raises:
            CustomException: 

        Returns:
            dict: model hyper paramerter dict 
        """
        try:           
            return self.all_params_dict.get(model_name)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def _return_model(self,model_name:str):
        """
        _return_model to return the non trained model based on model name

        Args:
            model_name (str): model name 

        Raises:
            CustomException: _description_
        """
        try:
            return self.all_models_dict.get(model_name)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def to_return_hyper_parameter_trained_model(self,model_name:str,x:pd.DataFrame,y:pd.DataFrame,base_accuracy=0.6):
        """
        to_return_hyper_parameter_trained_model to return best hyper parameter trained model 

        Args:
            model_name (str): trained model name
            x (pd.DataFrame): x_train data
            y (pd.DataFrame): y_train data
            base_accuracy (float, optional): base accuracy for trained model score. Defaults to 0.6.

        Raises:
            CustomException: 

        """
        try:
            model_params=self._return_params(model_name)
            model=self._return_model(model_name)
            random=RandomizedSearchCV(model,model_params)
            random.fit(x,y)
            # if random.best_score_ >= base_accuracy:
            return random.best_estimator_
            # return False
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    
  

class ToClassifyDataUsingCluster:
  def __init__(self):
    pass
    """
      ToClassifyDataUsingCluster to cluster the data 
    """
  def to_classify_data(self,df:pd.DataFrame,model_file_path:str,n_clusters=4)->pd.DataFrame:
    """
      to_classify_data to cluster tha data 

      Args:
          df (pd.DataFrame): data for cluster
          model_file_path (str): file path for cluster model save
          n_clusters (int, optional): no of groups to classify data . Defaults to 4.

      Raises:
          CustomException: 

      Returns:
          pd.DataFrame: to cliassified data 
    """
    try:
        new_df=df.copy()
        if OUT_COME_COLUMN_NAME in list(new_df.columns):
            new_df_1=new_df.drop(columns=OUT_COME_COLUMN_NAME)
        kmeans=KMeans(n_clusters=n_clusters,n_init=1)
        new_df['cluster_no']=kmeans.fit_predict(new_df_1)
        with open(model_file_path,'wb') as pickle_file:
            pickle.dump(  kmeans,pickle_file )
        return new_df
    except Exception as e:
      raise CustomException(error_msg=e, error_details=sys)

  def predict_data(self,predicted_data:pd.DataFrame,model_path:str)->pd.DataFrame:
    """
      predict_data to predict data on which cluster 

      Args:
          predicted_data (pd.DataFrame): predicted data 
          model_path (str): trained cluster model path

      Raises:
          CustomException: 

      Returns:
          pd.DataFrame: predicted data with cluster number
    """
    try:
        predicted_data_copy=predicted_data.copy()
        if OUT_COME_COLUMN_NAME in list(predicted_data_copy.columns):
            predicted_data_copy_1=predicted_data.drop(columns=OUT_COME_COLUMN_NAME)
        with open(model_path,'rb') as picle_file:
            model=pickle.load(picle_file)
        predicted_data_copy['cluster_no']=model.predict(predicted_data_copy_1)
        return predicted_data_copy
    except Exception as e:
      raise CustomException(error_msg=e, error_details=sys)



class CombineAll(ToClassifyDataUsingCluster,ReturnParams):

    def __init__(self,all_model_names_list:List[str]):
        """
        CombineAll to combine all classes

        Args:
            all_model_names_list (List[str]): all model name list
        """
        self.all_model_names=all_model_names_list
        self.all_params_dict=all_params_dict
        self.all_models_dict=all_models_dict

    def write_json(self,json_training_info_file_path:str,
                    content_key:str,content_value:str):
        """
        write_json to write a content into json file

        Args:
            json_training_info_file_path (str): json file path 
            content_key (str): content of key
            content_value (str): value of content
        """
        try:
            already_exist_content_dic=dict()
            if  os.path.exists(json_training_info_file_path):
                
                with open(json_training_info_file_path) as json_file:
                    already_exist_content_dic=json.load(json_file)
            with open(json_training_info_file_path,'w') as json_file:
                already_exist_content_dic.update({content_key:content_value})
                json.dump(already_exist_content_dic,json_file)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
    
    def save_best_model(self,model,model_path:str):
        # try:
        with open(model_path,'wb') as pickle_file:
            pickle.dump(model,pickle_file)

    def to_return_best_model(self,df:pd.DataFrame,target_col_name:str,to_stote_model_path:str,test_data:pd.DataFrame,
                          json_training_info_file_path:str,cluster_file_path:str,overall_model_info_json_file_path:str,
                          n_clusters=1,base_accuracy=0.6)->ModelTrainingArtifacts:
        """
        to_return_best_model to return best model score model

        Note :base_score accept for only model test score

        Args:
            df (pd.DataFrame): data 
            target_col_name (str): target column name
            to_stote_model_path (str): path to store the trained cluster model
            test_data (pd.DataFrame): test data
            json_training_info_file_path (str): json file path for store all training model info
            cluster_file_path (str): cluster file path
            overall_model_info_json_file_path (str): json file path for over all model info json
            n_clusters (int, optional): no of cluster classify data. Defaults to 1.
            base_accuracy (float, optional): min test accracy for trained model. Defaults to 0.6.

        Raises:
            CustomException: 

        Returns:
            ModelTrainingArtifacts: all trained model path
        """
        try:
            cluster_data=self.to_classify_data(df,cluster_file_path,n_clusters=n_clusters).groupby('cluster_no')
            test_df=self.predict_data(test_data,cluster_file_path)
            cluster_grp_test=test_df.groupby('cluster_no')
            
            
            model_dict_based_grp={}
            for grp,data in cluster_data:

                model_check_eligible_or_not_list=[]
                indivisual_model_dict={}
                for model_name in self.all_model_names:
                    train_score,test_score=0.0,0.0
                    x_train=data.drop(columns=[target_col_name,'cluster_no'])
                    y_train=data[target_col_name]

                    trained_model=self.to_return_hyper_parameter_trained_model(model_name=model_name,x=x_train,y=y_train,base_accuracy=base_accuracy)
                    if trained_model==False:model_check_eligible_or_not_list.append(trained_model)
                    train_score=trained_model.score(x_train,y_train)
                    if grp in test_df['cluster_no'].unique():
                        test_=test_df[test_df['cluster_no']==grp]
                    test_x=test_.drop(columns=['cluster_no',target_col_name])
                    test_y=test_[target_col_name]
                    test_score=trained_model.score(test_x,test_y)
                    indivisual_model_dict.update({
                        (train_score,test_score):
                        trained_model
                    })
                # if len(model_check_eligible_or_not_list)==len(all_model_names):
                #     return f"grp_{grp} data don't fit any model"
            
                demo_dict={ te_sc:tr_sc for tr_sc,te_sc in indivisual_model_dict.keys() }
                best_score=[(demo_dict.get(val),val)  for val in sorted(list(demo_dict.keys()),reverse=True)][0]
                best_model_=indivisual_model_dict.get(best_score)
                model_name=f"group_{grp}_{str(best_model_).split('(')[0]}.pkl"
                model_dict_based_grp.update( {(f'{model_name}',best_score): best_model_}  )

            for (key,val),model in model_dict_based_grp.items():
                

                model_path=os.path.join(to_stote_model_path,key)
                self.write_json(json_training_info_file_path=json_training_info_file_path,
                                content_key=model_path, content_value=list(val))

                self.write_json(json_training_info_file_path=overall_model_info_json_file_path,
                                content_key=model_path, content_value=list(val))
                                
                self.save_best_model(model=model,model_path=model_path)
            return ModelTrainingArtifacts(ovel_all_model_training_json_file_path=overall_model_info_json_file_path,saved_model_dir_path=to_stote_model_path)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
