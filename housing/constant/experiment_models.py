from lightgbm import LGBMRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

all_models_dict = {
    "lgbmregressor": LGBMRegressor(),
    "adaboostregressor": AdaBoostRegressor(),
    "gradientboostingregressor": GradientBoostingRegressor(),
    "randomforestgressor": RandomForestRegressor(),
    "decisontreegressor": DecisionTreeRegressor(),
    "lasso": Lasso(),
    "ridge": Ridge(),
    "elasticnet": ElasticNet(),
    "xgboost": XGBRegressor(),
}
