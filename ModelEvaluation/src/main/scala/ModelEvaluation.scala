import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

/**
  * Created by shijian on 20/05/2017.
  */
object ModelEvaluation {

  val spark = SparkSession.builder().appName("Spark").config("spark.master", "local").getOrCreate()

  val data = spark.read
    .format("libsvm").load("./data/sample_linear_regression_data.txt")

  // TRAIN TEST SPLIT
  val Array(training,testing) = data.randomSplit(Array(0.9, 0.1), seed=12345)

  data.printSchema()

  val lr = new LinearRegression()

  val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0.1, 0.01))
    .addGrid(lr.fitIntercept)
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .build()

  val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(lr)
    .setEvaluator(new RegressionEvaluator())
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)

  val model = trainValidationSplit.fit(training)

  model.transform(testing).select("features", "label", "prediction").show()
}
