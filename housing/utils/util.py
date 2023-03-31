from housing.exception import CustomException
import yaml
import os, sys
from housing.logger import logging


def read_yaml(file_path: str) -> dict:
    """
    this function toreading yaml by given path
    Args:
        file_path (str): yaml file

    Returns:
        dict: content of yaml file
    """
    try:
        all_info = None
        with open(file=file_path) as file:
            all_info = yaml.safe_load(file)
        return all_info
    except Exception as e:
        # excep=CustomException(error_msg=e, error_details=sys)
        # logging.error(msg=excep)
        raise CustomException(error_msg=e, error_details=sys) from e
