import org.apache.spark.{SparkConf, SparkContext}

import scala.util.parsing.json.JSON

object JSONRead {
  def main(args: Array[String]): Unit = {
    val inputFile = "E://idea/scalaPomProject/src/main/sources/people.json"
    val conf = new SparkConf().setAppName("JSONRead").setMaster("local")
    val sc = new SparkContext(conf)
    val jsonStrs = sc.textFile(inputFile)
    val result =jsonStrs.map(x=>JSON.parseFull(x))
    result.foreach({
      r=>r match{
        case Some(map:Map[String,Any])=>println(map)
        case None =>println("Parsing failed")
        case other => println("Unknown data structure:"+other)
      }
    })
  }

}
