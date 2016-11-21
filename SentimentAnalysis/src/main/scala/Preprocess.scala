import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.sql.SparkSession
/**
  * Created by Jackey on 16/11/20.
  */
object Preprocess {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Test Spark")
      .set("spark.executor.memory","2g")

    val sc = new SparkContext(conf)

    val posOriginData = sc.textFile("/Users/Jackey/Documents/datatest/movie/aclImdb/train/pos")
    val negOriginData = sc.textFile("/Users/Jackey/Documents/datatest/movie/aclImdb/train/neg")

    val posOriginDistinctData = posOriginData.distinct()
    val negOriginDistinctData = negOriginData.distinct()

//    val posAllRateDocument = posOriginDistinctData.map(line => line.split("\t"))
//    val negRateDocument = negOriginDistinctData.map(line => line.split("\t"))

    val posAllRateDocument = posOriginDistinctData
    val negRateDocument = negOriginDistinctData

    negRateDocument.repartition(1)
    val posRateDocument = sc.parallelize(posAllRateDocument.take(negRateDocument.count().toInt)).repartition(1)

    val posPair = posRateDocument.map(x => (x,1))
    val negPair = negRateDocument.map(x => (x,0))

    val allPair = negPair.union(posPair).repartition(1)

    val rate = allPair.values
    val document = allPair.keys


    val words = document.map(line => line.split(" ").toSeq)

//    val spark = new SparkSession()
//    val sentenceData = spark.createDataFrame(allPair).toDF("label","sentence")
//    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
//    val words2 = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(words)
    tf.cache()

    val idfObj = new IDF()
    val idfModel = idfObj.fit(tf)
    val tfidf = idfModel.transform(tf)

//    val rdd_idf = tfidf.map(row => (row(1), row(2)))
    val zipped = rate.zip(tfidf)
//    val zipped2 = tfidf.map(row => (rate, tfidf.rdd))
    val data = zipped.map(x => LabeledPoint(x._1, x._2))

    val Array(training,test) = data.randomSplit(Array(0.6, 0.4), 0)

    val NBmodel = NaiveBayes.train(training, 1.0)

    val predictionAndLabel = test.map(p => (NBmodel.predict(p.features), p.label))

    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println(accuracy)


  }
}
