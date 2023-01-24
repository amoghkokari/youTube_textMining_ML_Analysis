from pandas import DataFrame
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

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


@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> {}:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """
    # Specify your transformation logic here
    df["text"]=df["text"].apply(clean)
    vectorizer = vectorize(df["text"])
    vectorizer.fit(df["text"])
    
    X = fit_vectorizer(vectorizer,df["text"])
    y = df["like_count_1"]

    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.20, random_state=42)

    return {"X_train":X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test}


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    # assert df is not None, 'The output is undefined'
