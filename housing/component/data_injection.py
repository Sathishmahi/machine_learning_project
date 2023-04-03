import os
import sys
import tarfile

import numpy as np
import pandas as pd
import pyarrow
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from housing.entity.artifacts_entity import DataInjectionArtifacts
from housing.entity.config_entity import DataIngestionConfig
from housing.exception import CustomException
from housing.logger import logging


class DataInjection:
    def __init__(self, data_injection_config: DataIngestionConfig):
        """
        this function to help Data injection 
        1.download data from online
        2.unzip data
        3.to separate  data into train and test

        Args:
            data_injection_config (DataIngestionConfig): all data needs to DataInjection
        """
        self.data_injection_config = data_injection_config

    def create_folder(self, dir_path: str):
        try:
            if os.path.isdir(dir_path) and remove_dir:
                os.removedirs(dir_name)
            os.makedirs(dir_path)
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def download_data(self) -> str:
        """
        this function helps to  download the data in online 

        Returns:
            str: downloaded file path(tar file path)
        """
        try:
            ## to download data
            downloaded_url = self.data_injection_config.dataset_download_url

            ## to store data
            tgz_data_dir = self.data_injection_config.tgz_download_dir
            if os.path.isdir(tgz_data_dir):
                os.removedirs(tgz_data_dir)
            os.makedirs(tgz_data_dir, exist_ok=True)

            # file_name=downloaded_url.split('/')[-1]
            housing_file_name = os.path.basename(downloaded_url)

            ## file path tgz file
            tgz_file_path = os.path.join(tgz_data_dir, housing_file_name)
            logging.info(
                f"downloading file from download url [{downloaded_url}] file path is  [{tgz_file_path}]"
            )
            print(f"tar file path {downloaded_url}")
            urllib.request.urlretrieve(downloaded_url, tgz_file_path)
            logging.info(f"downloading file complete file path is  [{tgz_file_path}]")
            return tgz_file_path

        except Exception as e:
            CustomException(error_msg=e, error_details=sys)

    def extract_zip(self, tgz_file_path: str) -> str:
        """
        this function help us to extract a downloaded tar file 

        Args:
            tgz_file_path (str): tar file path

        Raises:
            CustomException: 

        Returns:
            str: extracted file path
        """
        try:
            raw_data_dir = self.data_injection_config.raw_data_dir
            self.create_folder(raw_data_dir)
            logging.info(
                f"extrct tar file start : tar file location is [{tgz_file_path}]"
            )
            with tarfile.open(tgz_file_path, "r:gz") as file:
                file.extractall(path=raw_data_dir)
            logging.info(
                f"extrct tar file finish : raw data location is [{raw_data_dir}]"
            )
            return raw_data_dir
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)

    def split_train_and_test(self) -> DataInjectionArtifacts:
        """
        split_train_and_test fuction help us to to split our overall data into train and test

        Raises:
            CustomException: 

        Returns:
            DataInjectionArtifacts: all created file path
        """
        try:
            raw_data_dir = self.data_injection_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            
            raw_data_file_path = os.path.join(raw_data_dir, file_name)
            housing_data_frame = pd.read_csv(raw_data_file_path)
            file_name=file_name.replace('csv', 'parquet')
            raw_data_file_path = os.path.join(raw_data_dir, file_name)
            housing_data_frame.to_parquet(raw_data_file_path,engine='pyarrow',index=False,compression='gzip')
            housing_data_frame=pd.read_parquet(raw_data_file_path)
            housing_data_frame["income_cat"] = pd.cut(
                housing_data_frame["median_income"],
                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=[1, 2, 3, 4, 5],
            )
            strat_train_data = None
            strat_test_data = None
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
            for train_index, test_index in split.split(
                housing_data_frame, housing_data_frame["income_cat"]
            ):
                strat_train_data = housing_data_frame.iloc[train_index, :].drop(
                    columns="income_cat"
                )
                strat_test_data = housing_data_frame.iloc[test_index, :].drop(
                    columns="income_cat"
                )
            train_data_dir = self.data_injection_config.ingested_train_dir
            self.create_folder(dir_path=train_data_dir)
            test_data_dir = self.data_injection_config.ingested_test_dir
            self.create_folder(dir_path=test_data_dir)

            train_file_path = os.path.join(train_data_dir, file_name)
            test_file_path = os.path.join(test_data_dir, file_name)

            if strat_test_data is not None and strat_train_data is not None:
                strat_test_data.to_parquet(test_file_path,engine='pyarrow',index=False,compression='gzip')
                strat_train_data.to_parquet(train_file_path,engine='pyarrow',index=False,compression='gzip')
            data_injecton_artifacts = DataInjectionArtifacts(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_injected=True,
                message=f"Data injected successfully",
            )
            return data_injecton_artifacts
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys) from e

    def initiate_data_injection(self) -> DataInjectionArtifacts:
        """
        initiate_data_injection function combine all data_injection functions 

        Raises:
            CustomException: 

        Returns:
            DataInjectionArtifacts: to return all data injection artifacts
        """
        try:
            tgz_file_path = self.download_data()
            raw_data_path = self.extract_zip(tgz_file_path)
            data_injection_arti = self.split_train_and_test()
            logging.info(
                msg=f"data injected successfully data injeciton artifacts  [{data_injection_arti}]"
            )
            return data_injection_arti
        except Exception as e:
            raise CustomException(error_msg=e, error_details=sys)
