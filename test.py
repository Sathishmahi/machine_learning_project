from typing import List
from housing.logger import logging
import sys
from housing.exception import CustomException
REQUIREMENTS_FILE_NAME='requirements.txt'
from housing.component.data_injection import DataInjection
from housing.pipeline.pipeline import Pipeline

p=Pipeline()
p.run_pipeline()