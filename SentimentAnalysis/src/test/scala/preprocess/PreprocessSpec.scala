package preprocess

import org.apache.spark
import org.apache.spark.rdd.RDD
import org.scalatest.{FlatSpec, Matchers}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util._

/**
  * Created by Jackey on 16/12/8.
  */
class PreprocessSpec extends FlatSpec with Matchers{
  behavior of "path2rdd"
  val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("Test Spark")
    .set("spark.executor.memory","2g")

  val sc = new SparkContext(conf)

  it should "work for valid path" in {

    val x = Preprocess.path2rdd("/Users/Jackey/Documents/datatest/movie/trainSample/pos",sc)

    x shouldBe a [RDD[String]]
  }

  it should "return non-empty rdd" in {

    val x = Preprocess.path2rdd("/Users/Jackey/Documents/datatest/movie/trainSample/pos",sc)

    x should not be empty
  }

  behavior of "makePair"
  val x = Preprocess.path2rdd("/Users/Jackey/Documents/datatest/movie/trainSample/pos",sc)
  x.distinct().repartition(1)

  it should "map correctly" in {
    val p = Preprocess.makePair(x)((e:RDD[String]) => e.map(v => (v,1)))
    p shouldBe a [RDD[(String,Int)]]
  }

  it should "give non-empty result" in {
    val p = Preprocess.makePair(x)((e:RDD[String]) => e.map(v => (v,1)))
    p should not be empty
  }

  it should "keys are string" in {
    val p = Preprocess.makePair(x)((e:RDD[String]) => e.map(v => (v,1)))
    p.keys shouldBe a [RDD[String]]
}
  it should "labels are int values " in {
    val p = Preprocess.makePair(x)((e:RDD[String]) => e.map(v => (v,1)))
    p.values shouldBe a [RDD[Int]]
  }

  it should "label positive review with 1" in {
    val p = Preprocess.makePair(x)((e:RDD[String]) => e.map(v => (v,1)))
    for(labels <- p.values.collect()) labels shouldBe 1
  }



}
