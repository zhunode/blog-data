import org.apache.spark.{SparkConf, SparkContext}

class rddPair {

}
/*
键值对rdd的创建
  1 第一种创建方式：1 从文件中加载
  可以采用多种方式创建Pari RDD,其中一种主要方式是使用map()函数来实现。

  2 第二种创建方式 通过并行集合(数组)创建RDD

* */
object rddPair{
  def main(args: Array[String]): Unit = {
    var sparkConf = new SparkConf().setAppName("rddPair").setMaster("local")
    var sc = new SparkContext(sparkConf)

    // 1 通过map()函数来实现pair rdd创建。
    var lines = sc.textFile("E://idea/myDemo/src/main/sources/word.txt")
    var pairRDD = lines.flatMap(line=>line.split(" ")).map(x=>(x,1))
    pairRDD.foreach(x=>println(x))

    // 2 通过并行集合(数组)创建RDD
    val list = List("Hadoop","Spark","Hive","Spark")
    val rdd = sc.parallelize(list)
    val pairRdd = rdd.map(word=>(word,1))
    pairRDD.foreach(x=>println(x))
    // 常用的键值对RDD转换操作
    // reduceByKey(func) 使用func函数合并具有相同键的值,
    pairRDD.reduceByKey((a,b)=>a+b).foreach(x=>println(x._1,x._2))

    // groupByKey() 对具有相同键的值进行分组
    pairRDD.groupByKey().foreach(x=>println(x))
    // 关于reduceByKey()和groupByKey()，尽量使用reduceByKey()方法
    println(pairRDD.values)

    // sortByKey()和sortBy()操作
    val d1 = sc.parallelize(Array(("c",8),("b",25),("c",17),("a",42),("b",4),("d",9),("e",17),("c",2),("f",29),("g",21),("c",9)))
    println(d1.reduceByKey((a,b)=>(a+b)).sortByKey(false).collect())

    d1.reduceByKey((x,y)=>x+y).sortBy(_._2,false).collect()

    // mapValues(func) 键值对RDD中的每一个value都应用一个函数，但是，key不会发生变化
    pairRDD.mapValues(x=>x+1).foreach(x=>println(x))

    // join就表示内连接。对于内连接，对于给定的两个输入数据集(K,V1)和(K,V2),只有在两个数据集中
    // 都存在的key才会被输出，最终得到一个(K,(V1,V2))类型的数据集。

    val pairRDD1 = sc.parallelize(Array(("spark",1),("spark",2),("hadoop",3),("hadoop",5)))
    val pairRDD2 = sc.parallelize(Array(("spark","fast")))

    pairRDD1.join(pairRDD2).foreach(x=>println(x))

    val rddtemp = sc.parallelize(Array(("spark",2),("hadoop",6),("hadoop",4),("spark",6)))
    rddtemp.mapValues(x=>(x,1)).reduceByKey((x,y)=>(x._1+y._1,x._2+y._2)).mapValues(x=>x._1/x._2).collect()





  }
}