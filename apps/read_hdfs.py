from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("Read CSV from HDFS") \
    .getOrCreate()

# Path to the CSV in HDFS
hdfs_path = "hdfs://namenode:9000/user/data/fraud.csv"

# Read the CSV into a DataFrame
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# Show the DataFrame
df.head(5)
