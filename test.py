from typing import List
from housing.logger import logging
import sys
import yaml
import pandas as pd
import numpy as np
from housing.exception import CustomException
REQUIREMENTS_FILE_NAME='requirements.txt'
from housing.component.data_injection import DataInjection
from housing.pipeline.pipeline import Pipeline

p=Pipeline(is_predicton=True)
p.run_pipeline()


# all_model_names_dict={"model_name_list":['lgbmregressor','adaboostregressor','gradientboostingregressor','randomforestgressor','decisontreegressor',
#     'lasso','ridge','elasticnet']}

# with open("/config/workspace/config/model.yaml",'w') as yaml_file:
#     yaml.safe_dump({"model_training":all_model_names_dict},yaml_file)