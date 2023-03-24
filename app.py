import flask
from flask import Flask,request,jsonify
app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return f'<h1>hello world</h1>'

if __name__=='__main__':
    app.run('0.0.0.0',debug=True)