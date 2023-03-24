from typing import List
from housing.logger import logging
import sys
from housing.exception import CustomException
REQUIREMENTS_FILE_NAME='requirements.txt'
    
try:
    20/0
except Exception as e:
    exe=CustomException(e,sys)
    print(exe)
    logging.error(exe)
def get_requirements_list()->List[str]:
    try:
        with open(REQUIREMENTS_FILE_NAME,'r') as file:
            logging.debug("error occured")
            print(file.readlines())
    except Exception as e:
        print(e)

# get_requirements_list()
