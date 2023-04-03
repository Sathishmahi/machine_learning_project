import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow
from sklearn.neighbors import KNeighborsRegressor

from housing.config.configuration import HousingConfig
from housing.constant import *
from housing.entity.artifacts_entity import (DataInjectionArtifacts,
                                             DataValidationArtifacts,
                                             FeatureEngineeringArtifacts)
from housing.exception import CustomException
from housing.logger import logging
from housing.utils import util
from housing.utils.util import read_yaml


class DataTransformation:

    def __init__(
        self,
        data_injection_artifacts: DataInjectionArtifacts,
        data_validation_artifacts: DataValidationArtifacts,
        config: HousingConfig = HousingConfig(),
        is_prediction_data=False,
        
        
    ):
        """
        DataTransformation to transform a data by using final report json 
        this report generate from EDA note book

        Args:
            data_injection_artifacts (DataInjectionArtifacts): DataInjectionArtifacts --  all file paths 
            data_validation_artifacts (DataValidationArtifacts): DataValidationArtifacts --- final report file path
            config (HousingConfig, optional): all config class. Defaults to HousingConfig().
            is_prediction_data (bool, optional): given data predicted or not. Defaults to False.

        Raises:
            CustomException: 
        """
        try:
            if not is_prediction_data:
                self.data_injection_artifacts = data_injection_artifacts
                self.data_transformation_config = config.model_transformation_config
                schema_file_path=config.data_validation_config.schema_file_path
                self.target_column=util.read_yaml(file_path=schema_file_path).get(TARGET_COLUMN_KEY)
                self.out_column_data=None
            # self.y_data=df.drop(columns=y_data)
        except Exception as e:
            raise CustomException(e, sys) from e

    def read_json(self, key: str,final_report_path="housing/component/final_report.json") -> dict:
        """
        read_json to read a final report json file 

        Args:
            key (str): key for json content 

        Returns:
            dict: to return the final report json content
        """
        # self.data_validation_artifacts.json_report_file_path
        with open(final_report_path, "r") as json_file:
            json_content = json.load(json_file)
        return json_content.get(key)

    def to_handle_cat_features(
        self,
        df: pd.DataFrame,
        transformed_train_data_dir: str,
        transformed_test_data_dir: str,
        json_file_path: str,
        file_name="to_handle_cat_features_train.parquet",
        thersold: int = 15,
        is_train_data=True,
        
    ):
        """
        to_handle_cat_features to handle the all categorical columns present in the data 

        Args:
            df (pd.DataFrame): data frame 
            transformed_train_data_dir (str): after handle the train data to store path
            transformed_test_data_dir (str): after handle the train data to store path
            json_file_path (str): json file to store all train transform cotents
            file_name (str, optional): fle for after handle cat features. Defaults to "to_handle_cat_features_train.parquet".
            thersold (int, optional): how many no of uniques to consier that columns as categorcal column. Defaults to 15.
            is_train_data (bool, optional): whether data was train or test data. Defaults to True.

        Raises:
            CustomException: 

        Returns:
            None
        """
        try:
            df_new = df.copy()
            all_dicreate_col_list = self.read_json(ALL_DISCRETE_COLUMNS_KEY)
            all_discrete_columns = [
                col for col in all_dicreate_col_list if col in df_new.columns
            ]
            mapped_dict = dict()
            if is_train_data:
                for column in all_discrete_columns:
                    idx = df_new[column].value_counts().index

                    if df_new[column].nunique() < thersold:
                        dic = dict(zip(idx, range(1, len(idx) + 1)))
                        mapped_dict.update({column: dic})
                        df_new[column] = df_new[column].map(dic)
                    else:
                        idx = df_new[column].unique()
                        dic = dict(zip(idx, range(1, len(idx) + 1)))
                        mapped_dict.update({column: dic})
                        mapped_dict.update({column: dic})
                        df_new[column] = df_new[column].map(dic)

                already_present_dict = dict()
                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as json_file:
                        already_present_dict = json.load(json_file)
                with open(json_file_path, "w") as json_file:
                    already_present_dict.update({HANDLE_CAT_FEATURES_DICT: mapped_dict})
                    json.dump(already_present_dict, json_file)
                file_path = os.path.join(transformed_train_data_dir, file_name)
                df_new.to_parquet(file_path,engine='pyarrow',index=False,compression='gzip')

                return file_path

            with open(json_file_path, "r") as json_file:
                json_content = json.load(json_file).get(HANDLE_CAT_FEATURES_DICT)
            df_new = df.copy()
            for column in all_discrete_columns:
                mapped_dic = json_content.get(column)
                df_new[column] = df_new[column].map(mapped_dic)
            file_name = "to_handle_cat_features_test.parquet"
            test_data_file_path = os.path.join(transformed_test_data_dir, file_name)
            df_new.to_parquet(test_data_file_path,engine='pyarrow', index=False,compression='gzip')

            return test_data_file_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def _helper_to_handle_na_values(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        model_path: str,
        model_save_or_not=False,
        k=5,
    ) -> np.array:
        """
        helper functioin for handle null values

        Args:
            x_train (pd.DataFrame): train x data 
            y_train (pd.DataFrame): train y data 
            x_test (pd.DataFrame): test x data 
            model_path (str): knn reg model path
            model_save_or_not (bool, optional): whether trained knn model save or not. Defaults to False.
            k (int, optional): no of neighbours(parameter of knn reg) . Defaults to 5.

        Raises:
            CustomException: 

        Returns:
            np.array: to return all prediction output
        """
        try:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(x_train, y_train)
            if model_save_or_not:
                with open(model_path, "wb") as pickle_file:
                    pickle.dump(knn, pickle_file)
            return knn.predict(x_test)
        except Exception as e:
            raise CustomException(e, sys) from e

    def to_handle_na_values(
        self,
        df: pd.DataFrame,
        model_save_dir: str,
        train_data_dir: str,
        test_data_dir: str,
        file_name="to_handle_na_values_train.parquet",
        is_train_data: bool = True,
    )->str:
        """
        to_handle_na_values to handle the all null value present in data 

        Args:
            df (pd.DataFrame): na present data 
            model_save_dir (str): knn reg model store path
            train_data_dir (str): after handle the na values to store a train data 
            test_data_dir (str): after handle the na vlaues dir to store a tets data 
            file_name (str, optional): after handle the na values file path. Defaults to "to_handle_na_values_train.parquet".
            is_train_data (bool, optional): whether given data train or test data. Defaults to True.

        Note: training data (all non null data consider as train data)

        Raises:
            CustomException: 

        Returns:
            str : path of after handle na values data 
        """
        try:
            if model_save_dir and train_data_dir:
                os.makedirs(model_save_dir, exist_ok=True)
                os.makedirs(train_data_dir, exist_ok=True)
            os.makedirs(test_data_dir, exist_ok=True)
            df_new = df.copy()
            all_na_columns_list = self.read_json(ALL_NULL_VALUES_COLUMNS_KEY)
            all_na_columns = [column[0] for column in all_na_columns_list]
            all_na_val_df = df_new[all_na_columns]
            non_na_columns = [
                col for col in df_new.columns if col not in all_na_columns
            ]
            all_non_na_val_df = df_new[non_na_columns]

            if is_train_data:
                save_or_not = True
                for na_column in all_na_columns:
                    model_file_path = os.path.join(model_save_dir, f"{na_column}.pkl")
                    na_idx = all_na_val_df[na_column][
                        all_na_val_df[na_column].isna()
                    ].index
                    non_na_idx = all_na_val_df[na_column][
                        all_na_val_df[na_column].isna() == False
                    ].index
                    prediction_values = self._helper_to_handle_na_values(
                        all_non_na_val_df.loc[non_na_idx],
                        all_na_val_df[na_column].loc[non_na_idx],
                        all_non_na_val_df.iloc[na_idx],
                        model_save_or_not=save_or_not,
                        model_path=model_file_path,
                    )
                    save_or_not = False
                    df_new[na_column].loc[na_idx] = prediction_values

                train_data_file_path = os.path.join(train_data_dir, file_name)
                df_new.to_parquet(train_data_file_path,engine='pyarrow', index=False,compression='gzip')

                return train_data_file_path

            for na_column in all_na_columns:
                na_idx = all_na_val_df[na_column][all_na_val_df[na_column].isna()].index
                non_na_idx = all_na_val_df[na_column][
                    all_na_val_df[na_column].isna() == False
                ].index
                trainded_model_path = os.listdir(model_save_dir)[0]
                print(f'trainded_model_path   {trainded_model_path}   model_save_dir    {model_save_dir} ')
                print('file path    ',os.path.join(model_save_dir, trainded_model_path))
                with open(
                    os.path.join(model_save_dir, trainded_model_path), "rb"
                ) as pickle_file:
                    trained_model = pickle.load(pickle_file)
                predicted_value = trained_model.predict(all_non_na_val_df.loc[na_idx])
                df_new[na_column].loc[na_idx] = predicted_value
            file_name = "to_handle_na_values_test.parquet"
            test_data_file_path = os.path.join(test_data_dir, file_name)
            df_new.to_parquet(test_data_file_path,engine='pyarrow', index=False,compression='gzip')
            
            return test_data_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def to_handle_mulitcolinerity(
        self,
        df: pd.DataFrame,
        model_save_dir: str,
        train_data_dir: str,
        test_data_dir: str,
        json_file_path: str,
        file_name="to_handle_mulitcolinerity_train.parquet",
        is_train_data: bool = True,
    ):
        try:
            # if self.is_prediction_data:
            
            
            df_new = df.copy()
            if self.target_column in list(df_new.columns):
                self.out_column_data=df_new[self.target_column]
                df_new=df_new.drop(columns=self.target_column)
            if is_train_data:
                os.makedirs(train_data_dir, exist_ok=True)
                all_multicolinearity_columns_keys = self.read_json(
                    ALL_MULTICOLINEARITY_COLUMNS_DICT_KEY
                )
                all_remove_columns_list = []
                for keys in all_multicolinearity_columns_keys:
                    column_1, column_2 = keys.split(" vs ")
                    if (column_1 not in all_remove_columns_list) and (
                        column_2 not in all_remove_columns_list
                    ):
                        all_remove_columns_list.append(column_1)
                all_remove_columns_list = [
                    col for col in all_remove_columns_list if col in df_new.columns
                ]
                df_new.drop(columns=all_remove_columns_list)
                train_file_path = os.path.join(train_data_dir, file_name)
                df_new.to_parquet(train_file_path, index=False,engine='pyarrow',compression='gzip')
                already_present_dict = dict()
                if os.path.exists(json_file_path):
            
                    with open(json_file_path, "r") as json_file:
                        already_present_dict = json.load(json_file)
                with open(json_file_path, "w") as json_file:
                    already_present_dict.update(
                        {
                            AFTER_HANDLE_THE_MULTICOLINEARITY_TRAIN_DF_COLUMNS_LIST: list(
                                df_new.columns
                            )
                        }
                    )
                    json.dump(already_present_dict, json_file)

                return train_file_path
            os.makedirs(test_data_dir, exist_ok=True)
            with open(json_file_path, "r") as json_file:
                all_multi_colinearity_columns = json.load(json_file).get(
                    AFTER_HANDLE_THE_MULTICOLINEARITY_TRAIN_DF_COLUMNS_LIST
                )
            all_remove_columns_list = [
                col for col in all_multi_colinearity_columns if col in df_new.columns
            ]
            df_new.drop(columns=all_remove_columns_list)
            
            file_name = "to_handle_mulitcolinerity_test.parquet"
            test_data_file_path = os.path.join(test_data_dir, file_name)
            df_new.to_parquet(test_data_file_path,engine='pyarrow', index=False,compression='gzip')
            
            return test_data_file_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def to_handle_negative_correlation(
        self,
        df: pd.DataFrame,
        model_save_dir: str,
        train_data_dir: str,
        test_data_dir: str,
        json_file_path: str,
        file_name="to_handle_negative_correlation_train.parquet",
        is_train_data: bool = True,
    )->str:
        """
        to_handle_negative_correlation to handle the all negative correlation columns (vs target column)
        

        Args:
            df (pd.DataFrame): data for hadle the neg corr
            model_save_dir (str): dir to store a model
            train_data_dir (str): dir to store after handle the neg corr train data
            test_data_dir (str): dir to store after handle the neg corr test data
            json_file_path (str): to store a all training data transformation info help us to handle the test data 
            file_name (str, optional): filename for after handle the neg corr data. Defaults to "to_handle_negative_correlation_train.parquet".
            is_train_data (bool, optional): whether given data train or test data. Defaults to True.

        Raises:
            CustomException: 

        Returns:
            str: after handle the neg corr data path
        """
        try:
            df_new = df.copy()
            if is_train_data:
                os.makedirs(train_data_dir, exist_ok=True)
                all_negative_corr_dict = self.read_json(
                    key=ALL_NEGATIVE_CORR_COLUMNS_KEY
                )
                all_removed_columns_list = [
                    columns
                    for columns in all_negative_corr_dict.keys()
                    if columns in df_new.columns
                ]
                df_new.drop(columns=all_removed_columns_list, inplace=True)
                train_file_path = os.path.join(train_data_dir, file_name)
                already_present_dict = dict()
                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as json_file:
                        already_present_dict = json.load(json_file)

                with open(json_file_path, "w") as json_file:
                    already_present_dict.update(
                        {
                            AFTER_HANDLE_NEGATIVE_CORRELATION_TRAIN_DF_COLUMNS_LIST: list(
                                df_new.columns
                            )
                        }
                    )
                    json.dump(already_present_dict, json_file)
                df_new.to_parquet(train_file_path,engine='pyarrow', index=False,compression='gzip')
                return train_file_path

            with open(json_file_path, "r") as json_file:
                all_remove_colunms_list = json.load(json_file).get(
                    AFTER_HANDLE_NEGATIVE_CORRELATION_TRAIN_DF_COLUMNS_LIST
                )
            df_new = df_new.loc[:, all_remove_colunms_list]
            file_name = "to_handle_negative_correlation_test.parquet"
            os.makedirs(test_data_dir, exist_ok=True)
            test_data_file_path = os.path.join(test_data_dir, file_name)
            df_new.to_parquet(test_data_file_path,engine='pyarrow', index=False,compression='gzip')
            return test_data_file_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def to_handle_non_normal_distribution(
        self,
        df: pd.DataFrame,
        model_save_dir: str,
        train_data_dir: str,
        test_data_dir: str,
        json_file_path: str,
        thersold: float = 2.0,
        file_name="to_handle_non_normal_distribution_train.parquet",
        is_train_data: bool = True,
    )->str:
        """
        to_handle_non_normal_distribution to handle the non normal distribution columns (like log-normal dist)

        Args:
            df (pd.DataFrame): data for handle the dist 
            model_save_dir (str): dir to store a model
            train_data_dir (str): dir to store after handle the neg corr train data
            test_data_dir (str): dir to store after handle the neg corr test data
            json_file_path (str): to store a all training data transformation info help us to handle the test data 
            thersold (float, optional): thersold for skewness of data  . Defaults to 2.0.
            file_name (str, optional): filename for after handle dist data . Defaults to "to_handle_non_normal_distribution_train.parquet".
            is_train_data (bool, optional): whether given data train or test data. Defaults to True.


        Raises:
            CustomException: 

        Returns:
            str: after handle dist of data file path
        """
        try:
            df_new = df.copy()
            if is_train_data:
                os.makedirs(train_data_dir, exist_ok=True)
                all_non_norml_distribution_list = self.read_json(
                    key=UNNORMAL_DIST_COLUMNS_KEY
                )
                all_non_norml_distribution_list = [
                    column
                    for column, skew_value in all_non_norml_distribution_list
                    if column in df_new.columns and skew_value > thersold
                ]
                for column in all_non_norml_distribution_list:
                    df_new[column] = np.log(df_new[column])
                train_file_path = os.path.join(train_data_dir, file_name)
                already_present_dict = dict()
                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as json_file:
                        already_present_dict = json.load(json_file)
                with open(json_file_path, "w") as json_file:
                    already_present_dict.update(
                        {
                            ALL_UNNORMAL_DISTRIBUTION_COLUMNS_LIST: all_non_norml_distribution_list
                        }
                    )
                    json.dump(already_present_dict, json_file)

                df_new.to_parquet(train_file_path,engine='pyarrow', index=False,compression='gzip')
                return train_file_path

            with open(json_file_path, "r") as json_file:
                os.makedirs(test_data_dir, exist_ok=True)
                all_transormation_columns = json.load(json_file).get(
                    ALL_UNNORMAL_DISTRIBUTION_COLUMNS_LIST
                )
            all_transormation_columns = [
                col for col in all_transormation_columns if col in df_new.columns
            ]
            for col in all_transormation_columns:
                df_new[col] = np.log(df_new[col])
            file_name = "to_handle_non_normal_distribution_test.parquet"
            test_file_path = os.path.join(test_data_dir, file_name)
            df_new.to_parquet(test_file_path,engine='pyarrow', index=False,compression='gzip')
            return test_file_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def to_remove_unwnated_columns(
        self,
        df: pd.DataFrame,
        train_data_dir: str,
        test_data_dir: str,
        json_info_file_path: str,
        file_name="final_data_train.parquet",
        is_train_data=True,
        is_predicted_data=False
    ) -> str:
        """
        to_remove_unwnated_columns to remove all unwnated columns like zero std columns 

        Args:
            df (pd.DataFrame): data for handle the dist 
            model_save_dir (str): dir to store a model
            train_data_dir (str): dir to store after handle the neg corr train data
            test_data_dir (str): dir to store after handle the neg corr test data
            json_file_path (str): to store a all training data transformation info help us to handle the test data 
            file_name (str, optional): filename for after handle dist data . Defaults to "final_data_train.parquet".
            is_train_data (bool, optional): whether given data train or test data. Defaults to True.
            is_predicted_data(bool, optional) : whether given data predicted data or not

        Raises:
            CustomException: 

        Returns:
            str: to return after remove the unwnated columns data 
        """
        try:
            after_transformed_data_path_list = []
            df_new=df.copy()
            if is_train_data:
                os.makedirs(train_data_dir, exist_ok=True)
                removed_column_list = [
                    col
                    for col in df_new.columns
                    if ("Unnamed" in col) or (np.std(df_new[col]) == 1.0)
                ]
              
                df_new.drop(columns=removed_column_list, inplace=True)
                train_file_path = os.path.join(train_data_dir, file_name)
                df_new[self.target_column]=self.out_column_data
                df_new.to_parquet(train_file_path, index=False,compression='gzip',engine='pyarrow')
                already_present_dict = dict()
                if os.path.exists(json_info_file_path):
                    with open(json_info_file_path) as json_file:
                        already_present_dict = json.load(json_file)
                with open(json_info_file_path, "w") as json_file:
                    already_present_dict.update(
                        {AFTER_REMOVE_ONE_STD_TRAIN_DIR_LIST: list(df_new.columns)}
                    )
                    json.dump(already_present_dict, json_file)
                return train_file_path
            with open(json_info_file_path, "r") as json_file:
                after_remove_one_std_columns_list = json.load(json_file).get(
                    AFTER_REMOVE_ONE_STD_TRAIN_DIR_LIST
                )
            after_remove_one_std_columns_list = [
                col
                for col in after_remove_one_std_columns_list
                if col in df_new.columns
            ]
            df_new = df_new.loc[:, after_remove_one_std_columns_list]
            file_name = "final_data_test.parquet"
            test_file_path = os.path.join(test_data_dir, file_name)
            if not is_predicted_data:
                df_new[self.target_column]=self.out_column_data
            df_new.to_parquet(test_file_path,engine='pyarrow', index=False,compression='gzip')
            return test_file_path

        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def initiate_data_transformation(self,saved_model_dir:str=None,latest_training_data_transformation_info_json_path:str=None,
                                    is_prediction_data=False,save_prediction_data_dir='prediction_data',prediction_df=None)->FeatureEngineeringArtifacts:
        
        """
        initiate_data_transformation to combine all functions

        Raises:
            CustomException: 

        Returns:
            FeatureEngineeringArtifacts: all file paths to generate this FeatureEngineering components
        """
        try:
            after_transformed_data_path_list=[]
            if not is_prediction_data:
                train_data_dir = self.data_transformation_config.transformed_train_dir
                test_data_dir = self.data_transformation_config.transformed_test_dir
                json_info_file_path = self.data_transformation_config.json_info_file_path
                saved_model_dir = (
                    self.data_transformation_config.preprocessed_object_file_path
                )
                train_file_path = self.data_injection_artifacts.train_file_path
                test_file_path = self.data_injection_artifacts.test_file_path
                
                df_train = pd.read_parquet(train_file_path)
                df_test = pd.read_parquet(test_file_path)
                combine_list = [[df_train, True], [df_test, False]]
                
                # df,is_train_data=combine_list[0][0],combine_list[0][1]
            else:
                train_data_dir = None
                test_data_dir = "save_prediction_data_dir"
                json_info_file_path = latest_training_data_transformation_info_json_path
                saved_model_dir = saved_model_dir
                train_file_path = None
                
                df_train =None
                df_test = prediction_df
                combine_list = [[df_test,False]]
            for df, is_train_data in combine_list:
                # with open(json_info_file_path ,'w') as f:
                #   json.dump({},f)
                multi_file_path = self.to_handle_mulitcolinerity(
                    df,
                    saved_model_dir,
                    train_data_dir,
                    test_data_dir,
                    json_info_file_path,
                    is_train_data=is_train_data,
                )
                print(f"finish multi colinerity")
                handle_negative_corr_path = self.to_handle_negative_correlation(
                    pd.read_parquet(multi_file_path),
                    saved_model_dir,
                    train_data_dir,
                    test_data_dir,
                    json_info_file_path,
                    is_train_data=is_train_data,
                )
                print(f"finish to handle negative correlation")
                handle_cat_columns_path = self.to_handle_cat_features(
                    pd.read_parquet(handle_negative_corr_path),
                    train_data_dir,
                    test_data_dir,
                    json_info_file_path,
                    is_train_data=is_train_data,
                )
                print(f"finish handle cat features")
                handle_na_values_path=handle_cat_columns_path
                df=pd.read_parquet(handle_cat_columns_path)
                print(df.isna().sum())
                if not is_prediction_data:
                    handle_na_values_path = self.to_handle_na_values(
                        df,
                        saved_model_dir,
                        train_data_dir,
                        test_data_dir,
                        is_train_data=is_train_data,
                    )
                    print("finish handle na values")
                after_handle_non_normal_dist_data_path = (
                    self.to_handle_non_normal_distribution(
                        pd.read_parquet(handle_na_values_path),
                        saved_model_dir,
                        train_data_dir,
                        test_data_dir,
                        json_info_file_path,
                        is_train_data=is_train_data,
                    )
                )

                print(f"finish to handle non normal dist")
                final_data_path = self.to_remove_unwnated_columns(
                    df=pd.read_parquet(after_handle_non_normal_dist_data_path),
                    train_data_dir=train_data_dir,
                    test_data_dir=test_data_dir,
                    json_info_file_path=json_info_file_path,
                    is_train_data=is_train_data,
                    is_predicted_data=is_prediction_data
                )
                print("finish data transformation")
                after_transformed_data_path_list.append(final_data_path)
            if not is_prediction_data:
                after_transformed_train_data_path, after_transformed_test_data_path = (
                    after_transformed_data_path_list[0],
                    after_transformed_data_path_list[1],
                )

                feature_engineering_artifacts = FeatureEngineeringArtifacts(
                    transformed_train_file_path=after_transformed_train_data_path,
                    transformed_test_file_path=after_transformed_test_data_path,
                    tranformation_model_path=saved_model_dir,
                    is_done_for_FE=True,
                    message=f"to finish feature enginnering",
                    trained_model_info_json_path=json_info_file_path
                )
                

                return feature_engineering_artifacts
            else:
                print(after_transformed_data_path_list)
                return after_transformed_data_path_list
            #     train_data_dir = None
            #     test_data_dir = save_prediction_data_dir
            #     json_info_file_path = self.data_transformation_config.json_info_file_path
            #     saved_model_dir = None
            #     train_file_path = None
                
            #     df_train =None
            #     df_test = prediction_df

        except Exception as e:
            raise CustomException(e, sys) from e
