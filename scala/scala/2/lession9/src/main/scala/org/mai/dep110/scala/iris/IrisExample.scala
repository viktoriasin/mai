package org.mai.dep110.scala.iris

import PetalSize.PetalSize

import scala.io.Source
import scala.util.Try

object IrisExample extends App {

  val flowers = loadFromFile("iris.data")
  println(flowers)

  //get average sepal width
  val avgSepalLength = flowers.map(p => p.sepalWidth).sum / flowers.length
  println(avgSepalLength)

  //get average petal square - petal width multiplied on petal length
  val avgPetalLength = flowers.map(p => p.sepalWidth * p.petalLength).sum / flowers.length
  println(avgPetalLength)

  //get average petal square for flowers with sepal width > 4
  val tuple = flowers
    .filter(p => p.sepalWidth > 4)
    .foldLeft(  Tuple2[Double, Int](0,0)) { (accumulator, iris) =>
      (accumulator._1 + iris.sepalWidth * iris.petalLength , accumulator._2+1)
    }
  val avgPetalSquare = tuple._1.toDouble/tuple._2.toDouble
  println(avgPetalSquare)

  //get flowers grouped by Petal size (PetalSize.Small, etc.) with function getPetalSize
  val groupsByPetalSize = flowers.groupBy(p => getPetalSize(p))
  println(groupsByPetalSize)
  println(groupsByPetalSize.keys)
  //get max sepal width for flowers grouped by species
  val maxSepalWidthForGroupsBySpecies = flowers.groupBy(p => p.species).mapValues( v => v.map(p => p.sepalWidth).max)
  println(maxSepalWidthForGroupsBySpecies)

//  val groupsByPetalSize1 = flowers.groupBy(p => getPetalSize(p)).

  def loadFromFile(path: String): List[Iris] = {
    Source
      .fromFile(path)
      .getLines
      .map(line => line.toIris)
      .filter{
        case Some(iris) => true
        case None => false
      }
      .map{
        case Some(iris) => iris
      }
      .toList
  }

  implicit class StringToIris(str: String) {
    def toIris: Option[Iris] = str.split(",") match {
      case Array(a,b,c,d,e) if isDouble(a) && isDouble(b) && isDouble(c) && isDouble(d) =>
        Some(
          Iris(
            a.toDouble,
            b.toDouble,
            c.toDouble,
            d.toDouble,
            e))
      case others => None
    }

    def isDouble(str: String): Boolean = Try(str.toDouble).isSuccess
  }

  def getPetalSize(iris: Iris): PetalSize = {
    val petalSquare = iris.petalLength * iris.petalWidth
    if(petalSquare < 2.0)
      PetalSize.Small
    if(petalSquare < 5.0)
      PetalSize.Medium
    else
      PetalSize.Large
  }

}

object PetalSize extends Enumeration {
  type PetalSize = Value
  val Large, Medium, Small = Value
}

case class Iris(sepalLength: Double, sepalWidth: Double, petalLength: Double, petalWidth: Double, species: String)
