import org.apache.spark.{SparkContext,SparkConf}
/**
  * Created by Jackey on 16/11/20.
  */
object WordCount {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Test Spark")
      .set("spark.executor.memory","2g")

    val sc = new SparkContext(conf)

    val lines = sc.parallelize(Seq("this is first line","this is second line","this is third line"))

    val counts = lines.flatMap(line => line.split(" "))
      .map(word => (word, 1))
      .reduceByKey(_+_)

    counts.foreach(println)
  }
}
