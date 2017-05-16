import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

import org.apache.log4j._

object Test {

  def main(args: Array[String]) {
    Logger.getLogger("com.example").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Spark").config("spark.master", "local").getOrCreate()

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv").load("./src/main/resources/Clean-USA-Housing.csv")

    data.printSchema()

  }
}

