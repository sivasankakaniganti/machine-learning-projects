from model_predict import model_prediction
from flask import Flask, render_template, request
from waitress import serve
app=Flask(__name__)
@app.route('/<text>')
def m1(text):
    val = model_prediction(text)
    return {'output':val}
print('serving port ',5333)
serve(app, host='0.0.0.0', port=5333)