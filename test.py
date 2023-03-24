from typing import List
import housing
REQUIREMENTS_FILE_NAME='requirements.txt'
def get_requirements_list()->List[str]:
    try:
        with open(REQUIREMENTS_FILE_NAME,'r') as file:
            print(file.readlines())
    except Exception as e:
        print(e)

get_requirements_list()
