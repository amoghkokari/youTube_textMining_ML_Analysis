from pyspark.sql.functions import col, concat, lit, udf, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF,StringIndexer
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

def process_model(data,algo,name):
    tokenizer=Tokenizer(inputCol="text",outputCol="words")
    remover=StopWordsRemover(inputCol="words",outputCol="filtered_words")
    countVec=CountVectorizer(inputCol="filtered_words",outputCol="features")
    idf=IDF(inputCol="features",outputCol="idf_features")
    label_stringIdx=StringIndexer(inputCol="like_count_1",outputCol="label_ind")

    #creating a pipeline
    pipeline=Pipeline(stages=[tokenizer,remover,countVec,idf,label_stringIdx,algo])

    #splitting the data
    (trainingData,testData)=data.randomSplit([0.8,0.2])

    #fitting the model
    model=pipeline.fit(trainingData)

    #predicting the results
    predictions=model.transform(testData)

    #evaluating the model
    evaluator_precision=MulticlassClassificationEvaluator(labelCol="label_ind",metricName="precisionByLabel")
    evaluator_recall=MulticlassClassificationEvaluator(labelCol="label_ind",metricName="recallByLabel")
    evaluator=MulticlassClassificationEvaluator(labelCol="label_ind",metricName="accuracy")
    evaluator_f1=MulticlassClassificationEvaluator(labelCol="label_ind",metricName="f1")

    dct = {"m_name":name}
    dct["precision"]=round(evaluator_precision.evaluate(predictions)*100, 2)
    dct["recall"]=round(evaluator_recall.evaluate(predictions)*100, 2)
    dct["accuracy"]=round(evaluator.evaluate(predictions)*100, 2)
    dct["f1"]=round(evaluator_f1.evaluate(predictions)*100, 2)

    return dct

def read_csv_spark(channel_id):
    spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()
    df = pd.read_csv("data/"+channel_id+".csv")
    df = df[['title','description','tags','like_count']]
    df = df.dropna(axis= 0, how= 'any')
    return spark.createDataFrame(df)

def process_spark_df(sdf):
    sf = udf(lambda x: x[1:-1])
    sdf1 = sdf.withColumn("tags_1", sf("tags"))
    sdf2 = sdf1.withColumn("text",concat(col("title"),lit(' '),col("tags_1"),lit(' '),col("description")))
    sdf3 = sdf2.select(['text', 'like_count'])
    sdf3 = sdf3.withColumn("like_count_1", when(col("like_count") <= sdf3.approxQuantile("like_count", [0.5], 0.25)[0], 0).otherwise(1))
    return sdf3

def spark_main(channel_id):
    sdf = read_csv_spark(channel_id)
    data = process_spark_df(sdf)
    lr=LogisticRegression(maxIter=10,regParam=0.3,labelCol="label_ind")
    lr_m = process_model(data,lr,"LogisticRegression")
    gbt = GBTClassifier(labelCol="label_ind", featuresCol="idf_features", maxIter=10)
    gbt_m = process_model(data,gbt,"GBTClassifier")
    nb = NaiveBayes(labelCol="label_ind", featuresCol="idf_features",smoothing=1.0, modelType="multinomial")
    nb_m = process_model(data,nb,"NaiveBayes")

    metrices_dct = {"LogisticRegression":lr_m,"GBTClassifier":gbt_m,"naiveBayes":nb_m}

    return metrices_dct