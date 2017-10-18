import java.util.logging.LogManager

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

object ReadCSVExample {

  def main(args: Array[String]): Unit = {

    val inputFile = args(0);

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate();

    import sparkSession.implicits;

    val patients = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .option("inferSchema", "true")
      .csv(inputFile)

    patients.show(100)
    patients.printSchema()

    val DFAssembler = new VectorAssembler().

      setInputCols(Array(
        "pregnancy", "glucose", "arterial pressure",
        "thickness of TC", "insulin", "body mass index",
        "heredity", "age")).
      setOutputCol("features")

    val features = DFAssembler.transform(patients)
    features.show(100)

    val labeledTransformer = {
      new StringIndexer().setInputCol("diabetes").setOutputCol("label")
    }
    val labeledFeatures = {
      labeledTransformer.fit(features).transform(features)
    }
    labeledFeatures.show(100)

    val splits = labeledFeatures.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)

    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.03)
      .setElasticNetParam(0.3)

    //Train Model
    val model = lr.fit(trainingData)

    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    //Make predictions on test data
    val predictions = model.transform(testData)
    predictions.show {
      200
    }

    //Evaluate the precision and recall
    val countProve = predictions.where("label == prediction").count()
    val countMistakes = predictions.where("label != prediction").count()
    val count = predictions.count()

    println(s"Count of true predictions: $countProve Total Count: $count \n Count of mistakes: $countMistakes Total Count: $count")
  }
}