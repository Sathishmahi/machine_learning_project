from typing import List
from housing.logger import logging
import sys
from housing.exception import CustomException
REQUIREMENTS_FILE_NAME='requirements.txt'
from housing.config.configuration import HousingConfig
from housing.component.data_injection import DataInjection

c=HousingConfig()
print(c.data_injection_config)
d=DataInjection(c.data_injection_config)
d.initiate_data_injection()