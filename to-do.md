# To Do

## Training all models using time, all features, and other resident activity 

### Go through each model one by one, and train it 

Fast Get good results tree_models
* decision_tree     "decision_tree": DecisionTreeModel,
* random_forest     "random_forest": RandomForestModel,


Ensemble models
* graidentboosting     "gradient_boosting": GradientBoostingModel,
* lightgbm    "lightgbm": LightGBMModel,(this one runs fast bad accuracy hyperparameter tuning)
* xgboost    "xgboost": XGBoostModel,
* catboost     "catboost": CatBoostModel,

they run but are not optimized, hyperparamater tuning is required


Dont work
* "knn": KNearestNeighborsModel,
* "svm": SupportVectorMachineModel,

knn doesnt work cause data is too large, computing distance betweeen each point could take  hours

svm doesnt work cause it runs infinitely, changed to sgdclassifier with hinge="Loss" to mimick linear svm (however Idk if it is appropriate for our problem)

we might have to scrap svm and do some polynomial sgd 

* needs to run for long time
"gaussiannb": GaussianNBModel,

* runs but its bad results
"logistic_regression": LogisticRegressionModel,



## Main approach

Two-fold comparison 

Comparison between raw off the shelf model training results
we can do it ourselves, we can optimize, 
or use the paper we are basing doing 

Feature enginereed same thing
all the shelf models with features faster reduces the amount
binning
time conversion
amount of activations iwthin 60 seconds, 5m, 10m
std
mean
normalize
kurtosis
coefficaint of variance
mean
apen


then find the best model that was featured engineered and train a deep learning model







