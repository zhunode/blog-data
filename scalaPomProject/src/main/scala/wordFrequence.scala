import org.apache.spark.{SparkConf, SparkContext}

object wordFrequence {
  def main(args: Array[String]): Unit = {
    var sparkConf = new SparkConf().setAppName("wordFrequence").setMaster("local")
    var sc = new SparkContext(sparkConf)
    var lines = sc.textFile("E://idea/myDemo/src/main/sources/word.txt")
    var wordCount = lines.flatMap(line=>line.split(" ")).map(x=>(x,1)).reduceByKey((a,b)=>a+b)
    wordCount.collect()
    wordCount.foreach(x=>println(x))
  }

}
