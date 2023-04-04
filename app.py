import flask
import pandas as pd
import json
import os,sys
from flask import Flask,request,jsonify,render_template,url_for
from housing.logger import logging
from housing.exception import CustomException
from housing.constant import PREDICTION_HELPER_JSON_FILE_NAME,OUT_COME_COLUMN_NAME
from housing.pipeline.pipeline import Pipeline
app=Flask(__name__)



def to_read_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME):
    try:
        with open(json_file_path) as json_file:
            return json.load(json_file)
    except Exception as e:
        raise CustomException(error_msg=e, error_details=sys)
@app.route('/',methods=['GET'])
def home():
    logging.info("logger start")
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict_():
    pipe=Pipeline(is_predicton=True)
    data=[float(val) for val in list(request.form.values())[:-1]]+[list(request.form.values())[-1]]
    train_file_path=to_read_json().get('data_injection_artifacts')[0]
    all_columns=list(pd.read_parquet(train_file_path).drop(columns=OUT_COME_COLUMN_NAME).columns)
    
    df=pd.DataFrame(dict(zip(all_columns,data)),index=[0])
    out_df=pipe.run_pipeline(prediction_data=df)
    return render_template('home.html', prediction_text=f"Median House Value is $ {int(out_df.out_come.values[0])}")

if __name__=='__main__':
    app.run('0.0.0.0',8000,debug=True) 