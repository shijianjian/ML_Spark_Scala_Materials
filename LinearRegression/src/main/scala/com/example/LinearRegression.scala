package com.example

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

import org.apache.log4j._

object Test {

  Logger.getLogger("com.example").setLevel(Level.ERROR)

  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("Spark").config("spark.master", "local").getOrCreate()

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv").load("./src/main/resources/Clean-USA-Housing.csv")

    /*
    *
    * Print out all all the fields of this data
    *
    * */
    data.printSchema()

    /*
    *
    * Print out all all the first col of data
    *
    * */
    val colnames = data.columns
    val firstrow = data.head(1)(0)
    println("\n")
    println("Example Data Row")
    for(ind <- Range(1, colnames.length)) {
      println(colnames(ind))
      println(firstrow(ind))
      print("\n")
    }

    // ("label", "features")
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors

    println(data.columns.mkString("\",\""))

    val df = data.select(data("Price").as("label"),
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

    val lr = new LinearRegression()

    val lrModel = lr.fit(output)

    val trainingSummary = lrModel.summary

    print(trainingSummary.residuals.show())
  }
}

