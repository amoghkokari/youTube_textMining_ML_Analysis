
from flask import Flask, render_template, request
import requests
app = Flask(__name__)
import urllib.request
from data_extraction import extract_data
from bs4 import BeautifulSoup

@app.route('/')
def student():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      res = process_result(result["ch1"])
      return res.head().to_html(classes='table table-stripped')
      
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
    res = parse(result)
    pro_df = extract_data(res)
    return pro_df

if __name__ == '__main__':
   app.run(debug = True)