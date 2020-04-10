class Test(val name:String) {
  private val id = Test.newId()
  def info(): Unit ={
    printf("This id of %s is %d.\n",name,id)
  }
}
object Test{
  private var lastId = 0
  def newId():Int ={
    lastId +=1
    lastId
  }

  def main(args: Array[String]): Unit = {
//    val person1 = new Test("junode")
//    val person2 = new Test("zhunode")
//    person1.info()
//    person2.info()

    // 针对容器的遍历操作
    val lis = List(1,2,3)
    val f_list = (i:Int)=>println(i)
    lis.foreach(f_list)
    // map 遍历
    val university = Map("XMU"->"zhunode","thu"->"t university")
    university.foreach(kv=>println(kv._1+" : "+kv._2))

    university foreach{x=>x match {case (k,v)=>println(k+":"+v)}} // unapply操作
    
  }
}