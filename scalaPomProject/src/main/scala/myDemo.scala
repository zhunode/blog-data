import org.apache.spark.{SparkConf, SparkContext}

object myDemo {
  // 计算每个键的平均值
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("average_test").setMaster("local")
    val sc = new SparkContext(conf)

    val add = sc.parallelize(Array(("spark",2),("hadoop",5),("spark",10),("hadoop",3)))
    val step1 = add.mapValues(x=>(x,1))
    step1.foreach(x=>println(x))
    println()

    val step2 = step1.reduceByKey((x,y)=>((x._1)+y._1,x._2+y._2))
    step2.foreach(x=>println(x))
    println()

    val step3 = step2.mapValues(x=>x._1/x._2)
    step3.foreach(x=>println(x))
  }

}
