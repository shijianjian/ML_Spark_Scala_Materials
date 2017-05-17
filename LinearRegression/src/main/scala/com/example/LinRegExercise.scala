package com.example

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

/**
  * Created by shijian on 17/05/2017.
  */
object LinRegExercise {

  val spark = SparkSession.builder().appName("Spark").config("spark.master", "local").getOrCreate()

  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv").load("./src/main/resources/Clean-Ecommerce.csv")

  data.printSchema()

  print(data.head(1).mkString(","))

  println(data.columns.mkString("\",\""))

  val df = data.select(
    data("Email"),
    data("Avatar"),
    data("Avg Session Length"),
    data("Time on App"),
    data("Time on Website"),
    data("Length of Membership"),
    data("Yearly Amount Spent").as("label"))

  val assembler = new VectorAssembler().setInputCols(Array(
    "Avg Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership")).setOutputCol("features")

  // Here, I presume that the Email and Avatar is not relevant to the Spent Money
  val output = assembler.transform(df).select("label", "Email", "Avatar", "features")

  val lr = new LinearRegression()

  // fit() is looking for the "label" & "features" field
  val lrModel = lr.fit(output)

  val coefficients = lrModel.coefficients

  val intercept = lrModel.intercept

  print("coefficients = " + coefficients + "; intercept = " + intercept)
}
