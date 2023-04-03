import flask
import pandas as pd
import json
from flask import Flask,request,jsonify,render_template,url_for
from housing.logger import logging
from housing.exception import CustomException
from housing.constant import PREDICTION_HELPER_JSON_FILE_NAME,OUT_COME_COLUMN_NAME
from housing.pipeline.pipeline import Pipeline
app=Flask(__name__)



def to_read_json(json_file_path=PREDICTION_HELPER_JSON_FILE_NAME):
    try:
        with open(PREDICTION_HELPER_JSON_FILE_NAME) as json_file:
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
    
    print(f'form value is {list(request.form.values())}')
    data=[float(val) for val in list(request.form.values())[:-1]]+[list(request.form.values())[-1]]
    
    train_file_path=to_read_json().get('data_injection_artifacts')[0]
    all_columns=list(pd.read_parquet(train_file_path).drop(columns=OUT_COME_COLUMN_NAME).columns)
    print(dict(zip(all_columns,data)))
    df=pd.DataFrame(dict(zip(all_columns,data)),index=[0])
    out_df=pipe.run_pipeline(prediction_data=df)
    out_df.out_come
    return render_template('home.html', prediction_text=f"Output is {out_df.out_come.values}")

if __name__=='__main__':
    app.run('0.0.0.0',8000,debug=True)