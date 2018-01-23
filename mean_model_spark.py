import pyspark as ps
from pyspark.sql import SparkSession
import dataclean
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.master("local[4]")\
        .appName("InteriorDesign")\
        .getOrCreate()
spark.conf.set("spark.executor.memory", "2g")
spark.conf.set("spark.driver.memory", "2g")

merged_df=dataclean.merge_data()
data=spark.createDataFrame(merged_df)
(trainingData, testData) = data.randomSplit([0.75, 0.25])

vectorAssembler = VectorAssembler(inputCols= columns,
                                      outputCol="features")

df_vector = vectorAssembler.transform(trainingData)
df_vector_test = vectorAssembler.transform(testData)

gbt = GBTRegressor(labelCol='rating',featuresCol='features',maxIter=500,step_size=0.05)
gbtmodel=gbt.fit(df_vector)
predictions = gbtmodel.transform(df_vector_test)

evaluator = RegressionEvaluator(
    labelCol="rating", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)
print("Mean Squared Error on test data = %g" % mse)
