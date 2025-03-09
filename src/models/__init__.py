from .tree_models import DecisionTreeModel, RandomForestModel
from .linear_models import LogisticRegressionModel, SupportVectorMachineModel

# from .ensemble_models import XGBoostModel, LightGBMModel, CatBoostModel
from .neighbors_models import KNearestNeighborsModel

# from .bayes_models import GaussianNBModel

# Model registry
MODEL_REGISTRY = {
    "knn": KNearestNeighborsModel,
    "svm": SupportVectorMachineModel,
    "decision_tree": DecisionTreeModel,
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    # "xgboost": XGBoostModel,
    # "lightgbm": LightGBMModel,
    # "catboost": CatBoostModel,
    # "gradient_boosting": GradientBoostingModel,
    # "gaussiannb": GaussianNBModel,
}


def get_model(model_name):
    """Factory function to get model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found")
    return MODEL_REGISTRY[model_name]()
