import os
import sys
from dataclasses import dataclass
import numpy as np
from src.logger import logging


from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                #"OneVsRest": OneVsRestClassifier(max_iter=500),
                "Logistic": LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=200),
                "svc": SVC(kernel='rbf', decision_function_shape='ovo'),
                "Decision Tree": DecisionTreeClassifier(),
                "RF classifier": RandomForestClassifier(n_estimators=100),
                "LGBM": LGBMClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(n_neighbors=5),
                #"xgb": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                "CatBoosting Classifier": CatBoostClassifier(verbose=0),
                "GaussianNB": GaussianNB(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                
            }
            params={
                "grid": {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'c_vals': [100, 10, 1.0, 0.1, 0.01],
                    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear']
                },
                "Random Forest":{
                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                     'max_features':['sqrt','log2',None],
                     'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            #logging.info("model report: ", model_report)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(best_model)

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            score = accuracy_score(y_test, predicted)
            return score


        except Exception as e:
            raise CustomException(e,sys)