all_params_dict = {
    "ridge": {"alpha": [0.001, 0.002, 0.003, 0.1, 0.2, 0.5, 1]},
    "lasso": {"alpha": [0.001, 0.002, 0.003, 0.1, 0.2, 0.5, 1]},
    "elasticnet": {
        "alpha": [0.001, 0.002, 0.003, 0.1, 0.2, 0.5, 1],
        "l1_ratio": [0.2, 0.3, 0.4, 0.5, 0.6],
    },
    "decisontreegressor": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "splitter": ["best", "random"],
        "min_samples_split": range(2, 10),
        "ccp_alpha": [0.001, 0.002, 0.01, 0.02, 0.1],
    },
    "randomforestgressor": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "min_samples_split": range(2, 10),
        "ccp_alpha": [0.001, 0.002, 0.01, 0.02, 0.1],
    },
    "gradientboostingregressor": {
        "learning_rate": [0.001, 0.002, 0.003, 0.01, 0.02, 0.1, 0.2],
        "n_estimators": range(30, 150, 20),
        "min_samples_split": range(2, 10),
        "n_iter_no_change": [5],
        "ccp_alpha": [0.001, 0.002, 0.01, 0.02, 0.1],
    },
    "adaboostregressor": {
        "learning_rate": [0.001, 0.002, 0.003, 0.01, 0.02, 0.1, 0.2],
        "n_estimators": range(30, 150, 20),
    },
    "lgbmregressor": {
        "learning_rate": [0.001, 0.002, 0.003, 0.01, 0.02, 0.1, 0.2],
        "min_split_gain": [0.001],
        "n_estimators": range(30, 150, 20),
    },
}
