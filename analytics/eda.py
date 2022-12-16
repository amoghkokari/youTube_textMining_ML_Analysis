import pandas as pd
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import os

def generate_wordcloud(df,key,channel_id):
    real_job=df[key].values
    wordcloud = WordCloud(width = 800, height = 800,background_color ='black',
        stopwords = STOPWORDS).generate(str(real_job))
    
    plt.switch_backend('Agg') 
    fig = plt.figure(figsize = (30,20))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    pname = key+'.png'
    plt.savefig("static/"+channel_id+"/"+pname)
    plt.close()
    return pname

def main(df, channel_id):
    df = pd.read_csv("data/"+channel_id+".csv")

    df1 = df.copy()

    df = df.dropna(axis= 0, how= 'any')

    df1['text'] = df1['title'] + df1['description']
    df1 = df1[['text','tags','like_count']]
    df1 = df1.dropna(axis= 0, how= 'any')
    df1["tags_1"]= df1["tags"].apply(lambda x: x[1:-1])
    df1['text']= df1['text'] + df1["tags_1"]
    df1["like_count_1"] = pd.qcut(df1["like_count"], 2, labels=[0,1]).astype("int64")
    df1 = df1.drop(['tags','like_count'], axis=1)

    df["tags_1"]= df["tags"].apply(lambda x: x[1:-1])
    df["like_count_1"] = pd.qcut(df["like_count"], 2, labels=[0,1]).astype("int64")

    if not os.path.exists("static/"):
        os.mkdir("static/")
    if not os.path.exists("static/"+channel_id+"/"):
        os.mkdir("static/"+channel_id+"/")

    tags_fig = generate_wordcloud(df,"tags_1",channel_id)
    title_fig = generate_wordcloud(df,"title",channel_id)
    desp_fig = generate_wordcloud(df,"description",channel_id)
    
    return df, df1, tags_fig, title_fig, desp_fig