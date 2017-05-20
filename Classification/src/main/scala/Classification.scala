import org.apache.log4j._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object Test {

//  Logger.getLogger("Classification").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("Spark").config("spark.master", "local").getOrCreate()

  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv").load("./data/train.csv")

  val test = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv").load("./data/test.csv")

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
//  val colnames = data.columns
//  val firstrow = data.head(1)(0)
//  println("\n")
//  println("Example Data Row")
//  for(ind <- Range(1, colnames.length)) {
//    println(colnames(ind))
//    println(firstrow(ind))
//    print("\n")
//  }

  // PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
  val logregdataall = data.select(
    data("Survived").as("label"),
    data("PassengerId"),
    data("Pclass"),
    data("Name"),
    data("Sex"),
    data("Age"),
    data("SibSp"),
    data("Parch"),
    data("Ticket"),
    data("Fare"),
    data("Cabin"),
    data("Embarked"))

  //PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
  val testSet = test.select(
    test("PassengerId"),
    test("Pclass"),
    test("Name"),
    test("Sex"),
    test("Age"),
    test("SibSp"),
    test("Parch"),
    test("Ticket"),
    test("Fare"),
    test("Cabin"),
    test("Embarked"))

  // Dropping rows containing any null values.
  val logregdata = logregdataall.na.drop()

  import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
  import org.apache.spark.ml.linalg.Vectors

  // Converting strings into numerical values
  val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
  val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

  // Convert Numberical Values into One Hot Encoding 0 or 1
  val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVector")
  val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVector")

  // (Label, features)
  val assembler = new VectorAssembler()
        .setInputCols(Array("Pclass", "SexVector", "Age", "SibSp", "Parch", "Fare", "EmbarkVector"))
        .setOutputCol("features")

  // Split the data set into 0.7 training, 0.3 testing
  // Seed is kind of splitting strategy, output will be same if fix the value of seed
   val Array(training, testing) = logregdata.randomSplit(Array(0.7, 0.3), seed=12345)

  import org.apache.spark.ml.Pipeline

  val lr = new LogisticRegression()

  val pipeline = new Pipeline()
      .setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, lr))

  val model = pipeline.fit(logregdata)

  val results = model.transform(testing)

  results.printSchema()

  //
  /////////////////////////
  // MODEL EVALUATION
  /////////////////////////
  import org.apache.spark.mllib.evaluation.MulticlassMetrics
  import spark.implicits._

  // Selection is only for label and features
  val predictionAndLabels = {
    results.select(results("prediction"), results("label")).as[(Double, Double)].rdd
  }

  val metrics = new MulticlassMetrics(predictionAndLabels)

  println("Confusion matrix: ")
  println(metrics.confusionMatrix)
}

