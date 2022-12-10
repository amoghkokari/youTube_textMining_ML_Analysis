from flask import Flask, render_template, request
import os
app = Flask(__name__)
import urllib.request
from data_extraction import extract_data
from bs4 import BeautifulSoup
from analytics.eda import main as eda_main
from analytics.preProcessing import main as pp_main
from analytics.classifiers import clf_main

@app.route('/')
def student():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      df, channel_id = process_result(result["ch1"])
      gt_viz, df = generate_visualizations(df,channel_id)
      ml_results = ml_classifiers(df)
      print(ml_results)
      hists = create_figUrl(channel_id)
      return render_template('result.html', hlst = hists)
      # return df.head().to_html(classes='table table-stripped')
      
def parse(res):
    page = urllib.request.urlopen(res)
    html = BeautifulSoup(page.read(),"html.parser")
    return html.find_all('meta',itemprop="channelId")[0].get('content')

# def get_html(df):
#     html = df.to_html()
#     text_file = open("templates/result.html", "w")
#     text_file.write(html)
#     text_file.close()

def process_result(result):
    channel_id = parse(result)
    pro_df = extract_data(channel_id)
    return pro_df, channel_id

def generate_visualizations(df,channel_id):
   df_1, tag, title, disc = eda_main(df,channel_id)
   sucess = 1
   return sucess, df_1

def ml_classifiers(df):
   X_train, X_test, y_train, y_test = pp_main(df)
   Tmodels, Fmetrices = clf_main(X_train, X_test, y_train, y_test)
   return Tmodels, Fmetrices

def create_figUrl(channel_id):
   hists = os.listdir('static/'+channel_id)
   hists = [file for file in hists]
   for i in range(len(hists)):
      hists[i] = 'static/'+channel_id+"/"+hists[i]
   return hists

if __name__ == '__main__':
   app.run(debug = True)