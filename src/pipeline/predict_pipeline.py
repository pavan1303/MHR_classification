import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Age: int,
        SystolicBP: int,
        DiastolicBP:int,
        BS: int,
        BodyTemp: int,
        ):

        self.Age = Age

        self.SystolicBP = SystolicBP

        self.DiastolicBP = DiastolicBP

        self.BS = BS

        self.BodyTemp = BodyTemp


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "SystolicBP": [self.SystolicBP],
                "DiastolicBP": [self.DiastolicBP],
                "BS": [self.BS],
                "BodyTemp": [self.BodyTemp],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
