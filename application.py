from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Age=float(request.form.get('age')),
            SystolicBP=float(request.form.get('systolicBP')),
            DiastolicBP=float(request.form.get('diastolicBP')),
            BS=float(request.form.get('bs')),
            BodyTemp=float(request.form.get('bodyTemp')),
            HeartRate=float(request.form.get('heartRate'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        prediction=predict_pipeline.predict(pred_df)
        result = prediction[0]
        if result == 0:
            result = 'low risk'
        if result == 1:
            result = 'mid risk'
        if result == 2:
            result = 'high risk'

        return render_template('home.html',results=result)


if __name__=="__main__":
    app.run(host="0.0.0.0")        
