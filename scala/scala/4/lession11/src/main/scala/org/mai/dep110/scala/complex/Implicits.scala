package org.mai.dep110.scala.complex

case class Complex[A: Arithmetic](re: A, im: A) {
  override def toString: String = s"$re+${im}i"

  def +(that: Complex[A]) : Complex[A] = {
    val num1 = implicitly[Arithmetic[A]].plus(this.re, that.re)
    val num2 = implicitly[Arithmetic[A]].plus(this.im, that.im)

    Complex(num1, num2)
  }
  def -(that: Complex[A]) : Complex[A] = {
    val num1 = implicitly[Arithmetic[A]].subtract(this.re, that.re)
    val num2 = implicitly[Arithmetic[A]].subtract(this.im, that.im)

    Complex(num1, num2)
  }
}

trait Arithmetic[A] {
  def n : A
  def plus(a:A, b:A): A
  def subtract(a:A, b:A): A

}
object Implicits {
  implicit object IntArithmetic extends Arithmetic[Int] {

    override def n: Int = 0

    override def plus(a: Int, b: Int): Int = a + b

    override def subtract(a: Int, b: Int): Int = a - b


  }

  implicit object LongArithmentic extends Arithmetic[Double] {

    override def n: Double = 0D

    override def plus(a: Double, b: Double): Double = a + b

    override def subtract(a: Double, b: Double): Double = a - b

  }

  implicit def TupleintToComplex[A: Arithmetic](n: (A, A)): Complex[A] = {
    Complex(n._1, n._2)
  }
//  implicit def TyplelongToComplex(n: (Double, Double)):  Complex[Double] = {
//    Complex(n._1, n._2)
//  }

  implicit def boundIntToComplex(a: Int) :  Complex[Int] = {
    Complex[Int](a,0)
  }

  implicit def boundlongToComplex(a: Double) :  Complex[Double] = {
    Complex[Double](a,0)
  }


  implicit class ComplexConversion[A: Arithmetic](a: A) {
    def real = Complex[A](a, implicitly[Arithmetic[A]].n)
    def imaginary = Complex[A](implicitly[Arithmetic[A]].n, a)

  }

}




