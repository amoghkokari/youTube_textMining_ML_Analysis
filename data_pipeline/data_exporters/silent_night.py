from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

def clean(text):
    text=text.lower()
    obj=re.compile(r"<.*?>")                     #removing html tags
    text=obj.sub(r" ",text)
    obj=re.compile(r"https://\S+|http://\S+")    #removing url
    text=obj.sub(r" ",text)
    obj=re.compile(r"[^\w\s]")                   #removing punctuations
    text=obj.sub(r" ",text)
    obj=re.compile(r"\d{1,}")                    #removing digits
    text=obj.sub(r" ",text)
    obj=re.compile(r"_+")                        #removing underscore
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s\w\s")                    #removing single character
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s{2,}")                    #removing multiple spaces
    text=obj.sub(r" ",text)
   
    
    stemmer = SnowballStemmer("english")
    stop=set(stopwords.words("english"))
    text=[stemmer.stem(word) for word in text.split() if word not in stop]
    
    return " ".join(text)

def vectorize(text):
    vectorizer=TfidfVectorizer(strip_accents='unicode',
                                analyzer='word',
                                ngram_range=(1, 2),
                                max_features=15000,
                                smooth_idf=True,
                                sublinear_tf=True)
    return vectorizer

def fit_vectorizer(vectorizer,text):
    return vectorizer.transform(text)

@data_exporter
def export_data(df: DataFrame, **kwargs):
    """
    Exports data to some source

    Args:
        df (DataFrame): Data frame to export to

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    df["text"]=df["text"].apply(clean)
    vectorizer = vectorize(df["text"])
    vectorizer.fit(df["text"])
    
    X = fit_vectorizer(vectorizer,df["text"])
    y = df["like_count_1"]

    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.20, random_state=42)

    model = GaussianNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    dct = {}

    dct["accuracy"] = round(accuracy_score(y_test,y_pred)*100,2)
    weighted_prf = precision_recall_fscore_support(y_test,y_pred,average='weighted')
    dct["precision"] = round(weighted_prf[0]*100, 2)
    dct["recall"] = round(weighted_prf[1]*100, 2)
    dct["f1score"] = round(weighted_prf[2]*100, 2)
    return dct
