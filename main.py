from flask import Flask, render_template, request
import time
import os
import cred
import pandas as pd
app = Flask(__name__)
import urllib.request
from data_extraction import extract_data
from bs4 import BeautifulSoup
from analytics.eda import main as eda_main
import analytics.preProcessing as pP
import analytics.spark_ml as spark_ml
from analytics.classifiers import clf_main
from analytics.classifiers import pred_success

@app.route('/')
def student():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   global channel_name
   if request.method == 'POST':
      exe_time = {}
      class_size = {}

      result = request.form
      df, channel_id = process_result(result["ch1"])
      channel_name=result.to_dict().get('ch1').split("/")[-1]
      gt_viz, df1, df = generate_visualizations(df,channel_id)

      ml_start_time = time.time()
      ml_results = ml_classifiers(df,channel_id)
      exe_time["pd_ml"]=(round(time.time()-ml_start_time,2))

      spark_start_time = time.time()
      ml_spark = spark_ml.spark_main(channel_id)
      exe_time["spark_ml"]=(round(time.time()-spark_start_time,2))

      class_size["low"] = len(df[df["like_count_1"]==0])
      class_size["high"] = len(df[df["like_count_1"]==1])
      hists = create_figUrl(channel_id)
      return render_template('result.html', hlst=hists, ml=ml_results, extime=exe_time, mlspark=ml_spark, dshape=df.shape, csize=class_size, channel_name=channel_name)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   preds = ""
   if request.method == 'POST':
      result = request.form
      text = result["title"] +" "+ result["desc"]
      ctext = pP.clean(text)
      pdf = pd.DataFrame(data={"text":[ctext]})
      X = pP.fit_vectorizer(cred.vect,pdf["text"])
      preds = pred_success(X.toarray(),cred.channel_id)
   return render_template('predict.html', predn = preds, channel_name=channel_name) 

def parse(res):
    page = urllib.request.urlopen(res)
    html = BeautifulSoup(page.read(),"html.parser")
    return html.find_all('meta',itemprop="channelId")[0].get('content')

def process_result(result):
    channel_id = parse(result)
    cred.channel_id = channel_id
    pro_df = extract_data(channel_id)
    return pro_df, channel_id

def generate_visualizations(df,channel_id):
   df, df_1, tag, title, disc = eda_main(df,channel_id)
   sucess = 1
   return sucess, df, df_1

def ml_classifiers(df,channel_id):
   X_train, X_test, y_train, y_test = pP.main(df)
   Tmodels, Fmetrices = clf_main(X_train, X_test, y_train, y_test,channel_id)
   return Fmetrices

def create_figUrl(channel_id):
   hists = os.listdir('static/'+channel_id)
   hists = [file for file in hists]
   for i in range(len(hists)):
      hists[i] = 'static/'+channel_id+"/"+hists[i]
   return hists

if __name__ == '__main__':
   app.run(debug = True)