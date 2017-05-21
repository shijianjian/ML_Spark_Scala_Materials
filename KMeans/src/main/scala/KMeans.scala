import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler

/**
  * Created by shijian on 20/05/2017.
  */
object KMeansClustering {

  val spark = SparkSession.builder().appName("Spark").config("spark.master", "local").getOrCreate()

  val dataset = spark.read
    .option("header","true")
    .option("inferSchema","true")
    .csv("./data/Wholesale_customers_data.csv")
//Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
  val feature_data = dataset.select(
    dataset("Fresh"),
    dataset("Milk"),
    dataset("Frozen"),
    dataset("Detergents_Paper"),
    dataset("Delicassen"))

  val assemble = new VectorAssembler()
    .setInputCols(Array("Fresh", "Milk", "Frozen", "Detergents_Paper", "Delicassen"))
    .setOutputCol("features")

  val training_data = assemble.transform(feature_data)

  // Trains a k-means model. K = 3 means 3 clusters.
  val kmeans = new KMeans().setK(3)
  val model = kmeans.fit(training_data)

  // Evaluate clustering by computing Within Set Sum of Squared Errors.
  val WSSSE = model.computeCost(training_data)

  // Shows the result.
  println(s"Within Set Sum of Squared Errors = $WSSSE")
}
