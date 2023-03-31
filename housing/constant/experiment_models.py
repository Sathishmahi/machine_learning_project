from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

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
