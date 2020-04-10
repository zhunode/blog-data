class classDemo {
  private var value = 0
  private var name = ""
  private var step = 1
  println("the main constructer")
  def this(name:String){
    this()// 调用主构造器
    this.name = name
    printf("this first auxiliary constructor,name:%s\n",name)
  }

  def this(name:String,step:Int){// 第二个构造器
    this(name)
    this.step = step
    printf("the second auxiliary constructor:%s,step:%d\n",name,step)
  }

  def increment(step:Int): Unit ={
    value+=step
  }
  def current(): Int ={
    value
  }

}
// 使用object关键字定义单例对象
// 单例对象的使用和一个普通的类实例一样。

object classDemo{
  def demo(step:Int): Int ={
    step+30
  }
  def main(args: Array[String]): Unit = {
    var c1 = new classDemo
    println(c1)

    c1 = new classDemo("the first Constructor")
    println(c1)

    c1 = new classDemo("the second constructor",3)
    println(c1.current())
    println(classDemo.demo(30))
  }
}