class scala_ArrayDemo {
  private var privateValue = 0
  def value = privateValue
  def value_=(newValue: Int){
    if (newValue > 0) privateValue = newValue
  }
  def increment(step: Int): Unit = {value += step}
  def current():Int = {value}
}


object scala_ArrayDemo {
  def main(args: Array[String]): Unit = {
    // 数组创建
    val intValueArr = new Array[Int](3) // 声明一个长度为3的整形数组，每个数组元素初始化为0.
    intValueArr(0)=11
    intValueArr(1)=45
    intValueArr(2)=33

    val myStrArr = new Array[String](3) // 声明一个长度为3的字符串数组，每个元素初始化为null
    myStrArr(0) ="BigData"
    myStrArr(1)="Hadoop"
    myStrArr(2)="Spark"

    // tuple元组创建
    val tuple1=("BigData",2015,11,30)
    println(tuple1._1)
    println(tuple1._4)

    val mycount = new scala_ArrayDemo
    mycount.increment(3)
    println(mycount.value)
    println(mycount.current())

  }
}
