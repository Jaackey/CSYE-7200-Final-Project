import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

/**
  * Created by Jackey on 16/12/6.
  */
object LogiReg {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Test Spark")
      .set("spark.executor.memory", "2g")

    val sc = new SparkContext(conf)

    //  val spark = new SparkSession()

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    //  val sc = new SparkContext()
    //  val sc = spark.sparkContext

//    val pos_base_path = "/Users/Jackey/Documents/datatest/movie/aclImdb/train/pos"
//    val neg_base_path = "/Users/Jackey/Documents/datatest/movie/aclImdb/train/neg"

    val pos_base_path = "/Users/Jackey/Documents/datatest/movie/trainSample/pos"
    val neg_base_path = "/Users/Jackey/Documents/datatest/movie/trainSample/neg"



    val raw_pos = sc.textFile(pos_base_path).map(x => (x, 1)).toDF("sentence", "label")
    val raw_neg = sc.textFile(neg_base_path).map(x => (x, 0)).toDF("sentence", "label")
    var raw = raw_pos.union(raw_neg)
    println("total training set: " + raw.count())

    // COMMAND ----------

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(raw)

    // COMMAND ----------

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("features", "label").take(3).foreach(println)

    // COMMAND ----------

    val Array(training, testing) = rescaledData.randomSplit(Array(0.8, 0.2))

    // COMMAND ----------

    import org.apache.spark.ml.classification.LogisticRegression


    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // COMMAND ----------

//    val predictions = lrModel.transform(testing)
//    val evaluator = new BinaryClassificationEvaluator()
//      .setLabelCol("label")
//      .setRawPredictionCol("sentence")
//    val accuracy = evaluator.evaluate(predictions)
//    println("Accuracy = " + accuracy)
    

  }

}
