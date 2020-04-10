import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object demo{
  def main(args: Array[String]): Unit = {1
//    val str = "hello world";
//    println(str)
//    val sparkConf = new SparkConf().setAppName("spark api").setMaster("local")
//    val sparkContext = new SparkContext(sparkConf)
//    val data = Array(1,2,3,4,5)
//    val inputRdd = sparkContext.parallelize(data);
//    inputRdd.collect().foreach(println(_))

    // RDD learning
    val sparkConf = new SparkConf().setAppName("helloTest").setMaster("local")
    val sc = new SparkContext(sparkConf)
//    val lines = sc.textFile("E://idea/myDemo/src/main/sources/word.txt")
//    val linesWithSpark = lines.filter(line => line.contains("Spark"))
//    linesWithSpark.foreach(x=>println(x))

    // map 操作：将每一个元素传递到函数func中，并将结果返回为一个新的数据集。
    val data = Array(1,2,3,4,5)
    val rdd1 = sc.parallelize(data)
    val rdd2 = rdd1.map(x=>x+10)
    rdd2.foreach(x=>println(x))
//    var words = lines.map(line=>line.split(" "))
//    var wordFlatmap = lines.flatMap(line=>line.split(" "))
//    var resultByKey = lines.flatMap(line=>line.split(" ")).map(x=>(x,1)).reduceByKey((a,b)=>a+b)
//    resultByKey.foreach(x=>println(x))
//    println(rdd1.count())// action操作的 count()计数
//    println(rdd1.first())// action操作 获取第一个元素
//    println(rdd1.take(3)) // action操作 获取前3个元素
//    println(rdd1.reduce((a,b)=>a+b))// action操作 元素累加操作
//    println(rdd1.foreach(elem=>println(elem)))
//    println(lines.map(x=>x.length))// 长度
//    println(lines.map(x=>x.length).reduce((a,b)=>a+b))

    // 持久化操作
    val list = List("Hadoop","Spark","Hive")
    val rdd = sc.parallelize(list)
    println(rdd.count())// Action操作，触发一次真正的从头到尾的计算
    println(rdd.collect().mkString(","))// Action操作，触发一次真正的从头到尾的计算。

    // 分区操作，使用repartition方法即可
//    val data2 = sc.textFile("E://idea/myDemo/src/main/sources/word.txt",2)
//    println(data2.partitions.size) // 显示data这个RDD的分区数量
//    val rdd3 = data2.repartition(1) // 对data2这个RDD进行重新分区
//    println(rdd3.partitions.size)
  }
}