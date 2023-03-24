from setuptools import setup,find_packages
from typing import List
## declering variable for setup function
PROJECT_NAME="housing-predictor"
VERSION="0.0.2"
AUTHOR="sathish"
DESCRIPTION="""
this is a housing price predictor 
"""
REQUIREMENTS_FILE_NAME='requirements.txt'


def get_requirements_list()->List[str]:

    """
    this func to return list of str by requirements.txt

    Returns:
        List[str]:all library present in requirements.txt
    """

    with open(REQUIREMENTS_FILE_NAME) as file:
        return file.readlines()

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages= find_packages(),#["housing"],
    install_requirements=get_requirements_list().remove('-e .')


)