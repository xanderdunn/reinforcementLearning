import org.scalatest._
import collection.mutable.Stack
import neuralNet._
import neuralNet.NeuralNetUtilities._

abstract class UnitSpec extends FlatSpec with Matchers with
  OptionValues with Inside with Inspectors

class ExampleSpec extends FlatSpec with Matchers {

  "A NeuralNet" should "correctly convert a state and action into a featureVector" in {
    var featureVetor = neuralNetFeatureVectorForStateAction(List("X", "", "", "", "", "", "" , "", ""))
    featureVetor should equal (Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    featureVetor = neuralNetFeatureVectorForStateAction(List("X", "", "", "O", "", "O", "" , "", ""))
    featureVetor should equal (Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0))
  }

  it should "be able to learn sin(x)" in {
    val neuralNet = new NeuralNet(1, 20, 0.05, 1.0)
    var i = 0
    while (i < 100000) { // Train
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      neuralNet.train(Array(x), y)
      i += 1
    }
    i = 0
    while (i < 1000) { // Test it works to a degree
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      val result = neuralNet.feedForward(Array(x))
      var withinRange = false
      if (result < y + 0.1 && result > y - 0.1) {
        withinRange = true
      }
      else {
        println(s"x = ${x}")
        println(s"result = ${result}")
      }
      withinRange should equal (true)
      i += 1
    }
    while (i < 1000) { // Negative test to check that the test itself isn't broken
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      val result = neuralNet.feedForward(Array(x))
      var withinRange = true
      if (result < y + 0.01 && result > y - 0.01) {
        withinRange = false
      }
      else {
        println(s"x = ${x}")
        println(s"result = ${result}")
      }
      withinRange should equal (false)
      i += 1
    }
  }

  it should "be able to learn x=y" in {
    val neuralNet = new NeuralNet(1, 10, 0.1, 1.0)
    var i = 0
    while (i < 100000) { // Train
      val x = scala.util.Random.nextDouble()
      //println(s"input = ${x}")
      val result = neuralNet.feedForward(Array(x))
      //println(s"result = ${result}")
      neuralNet.train(Array(x), x)
      i += 1
    }
    i = 0
    while (i < 1000) {
      val x = scala.util.Random.nextDouble()
      val result = neuralNet.feedForward(Array(x))
      var withinRange = false
      if (result < x + 0.1 && result > x - 0.1) {
        withinRange = true
      }
      else {
        println(s"x = ${x}")
        println(s"result = ${result}")
      }
      withinRange should equal (true)
      i += 1
    }
  }
}
