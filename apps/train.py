from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    # Create Spark Session
    # spark = SparkSession.builder \
    #     .appName("FraudDetection") \
    #     .config("spark.master", "spark://localhost:7077") \
    #     .config("spark.submit.deployMode", "client") \
    #     .getOrCreate()
    spark = SparkSession.builder \
        .appName("FraudDetection") \
        .config("spark.master", "spark://localhost:7077") \
        .config("spark.submit.deployMode", "client") \
        .getOrCreate()

    # Define schema for the dataset
    schema = StructType([
        StructField("step", IntegerType(), True),
        StructField("type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("nameOrig", StringType(), True),
        StructField("oldbalanceOrg", DoubleType(), True),
        StructField("newbalanceOrig", DoubleType(), True),
        StructField("nameDest", StringType(), True),
        StructField("oldbalanceDest", DoubleType(), True),
        StructField("newbalanceDest", DoubleType(), True),
        StructField("isFraud", IntegerType(), True)
    ])

    # Load data from HDFS
    df = spark.read.csv("hdfs://namenode:8020/data/fraudt.csv", header=True, schema=schema)

    # Preprocess data
    indexer = StringIndexer(inputCol="type", outputCol="typeIndex")
    encoder = OneHotEncoder(inputCol="typeIndex", outputCol="typeVec")
    assembler = VectorAssembler(
        inputCols=["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "typeVec"],
        outputCol="features")

    pipeline = Pipeline(stages=[indexer, encoder, assembler])
    df_prepped = pipeline.fit(df).transform(df)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(labelCol="isFraud", featuresCol="features"),
        "RandomForest": RandomForestClassifier(labelCol="isFraud", featuresCol="features"),
        "GBTClassifier": GBTClassifier(labelCol="isFraud", featuresCol="features")
    }

    # Train, predict, and evaluate each model
    results = {}
    for name, model in models.items():
        model_fitted = model.fit(df_prepped)
        predictions = model_fitted.transform(df_prepped)
        evaluator = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="prediction")
        auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        results[name] = auc
        print(f"{name} - AUC: {auc}")

    # Stop Spark Session
    spark.stop()