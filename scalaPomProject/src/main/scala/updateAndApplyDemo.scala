class updateAndApplyDemo(name:String) {
  // 关于 val myStrarr = Array("BigData","Hadoop","Spark")的执行过程
  def info(): Unit ={
    println("Car name is "+name)
  }
}

object updateAndApplyDemo{
  // 调用伴声类Car的构造方法
  def apply(name: String): updateAndApplyDemo = new updateAndApplyDemo(name)
}

object MyTestApply{
  def main(args: Array[String]): Unit = {
    val car = updateAndApplyDemo("BMW") // 调用伴生对象中的apply方法
    car.info()

    // 遍历操作
    var list = List(1,2,3)
    list.foreach((x)=>println(x))

    var university = Map("XMU"->"厦门大学","gvd"->"广州大学")
    university.foreach((kv)=>println(kv._2,kv._1))

    // 映射操作
    val books = List("Hadoop","Hive","HDFS")
    println(books.map(i=>i.toUpperCase()))

    println(books.map(i=>i.length))

    // flatmap扁平操作
    println(books.flatMap(i=>i.toList))

    // 过滤操作
    println(university.filter(x=>x._2.contains("大")))
  }
}