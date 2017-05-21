import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._
/**
  * Created by shijian on 20/05/2017.
  */
object RegressionModelEvaluation {

  Logger.getLogger("com.example").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark").config("spark.master", "local").getOrCreate()

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv").load("./data/Clean-USA-Housing.csv")

    data.printSchema()

    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors

    val df = data.select(
      data("Price").as("label"),
      data("Avg Area Income"),
      data("Avg Area House Age"),
      data("Avg Area Number of Rooms"),
      data("Avg Area Number of Bedrooms"),
      data("Area Population"))

    // features will be array of 5 features
    val assembler = new VectorAssembler().setInputCols(Array(
      "Avg Area Income",
      "Avg Area House Age",
      "Avg Area Number of Rooms",
      "Avg Area Number of Bedrooms",
      "Area Population")).setOutputCol("features")

    val output = assembler.transform(df).select("label", "features")


    //// TRAINING AND TEST data
    val Array(training, testing) = output
      .select("label", "features")
      .randomSplit(Array(0.7, 0.3))

    // model
    val lr = new LinearRegression()

    // Parameter Grid builder
    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(100000000, 0.00001)).build()

    // Train Split
    val trainvalsplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    val model = trainvalsplit.fit(training)

    model.transform(testing).select("features", "label", "prediction").show()
}
