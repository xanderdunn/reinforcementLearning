import org.scalatest.{FlatSpec, Matchers}
import tags.{CoverageAcceptanceTest, NonCoverageAcceptanceTest, UnitTest}
import neuralNet.MultiLayerPerceptron
import activationFunctions.{TangentSigmoidActivationFunction, LinearActivationFunction}

class NeuralNetSpec extends FlatSpec with Matchers {
  "A NeuralNet" should "be able to learn sin(x)" taggedAs(CoverageAcceptanceTest) in {
    val neuralNet = new MultiLayerPerceptron(List(1, 20, 1), List(new TangentSigmoidActivationFunction(), new LinearActivationFunction()))
    var i = 0
    while (i < 100000) { // Train
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      neuralNet.train(List(x), List(y))
      i += 1
    }
    i = 0
    while (i < 1000) {
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      val result = neuralNet.feedForward(List(x))(0)
      result should equal (y +- 0.1)
      i += 1
    }
    while (i < 1000) { // Negative test to check that the test itself isn't broken
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      val result = neuralNet.feedForward(List(x))(0)
      result should not equal (y +- 0.01)
      i += 1
    }
  }

  it should "be able to learn x=y" taggedAs(CoverageAcceptanceTest) in {
    val neuralNet = new MultiLayerPerceptron(List(1, 10, 1), List(new TangentSigmoidActivationFunction(), new LinearActivationFunction()))
    var i = 0
    while (i < 100000) { // Train
      val x = scala.util.Random.nextDouble()
      val result = neuralNet.feedForward(List(x))
      neuralNet.train(List(x), List(x))
      i += 1
    }
    i = 0
    while (i < 1000) {
      val x = scala.util.Random.nextDouble()
      val result = neuralNet.feedForward(List(x))(0)
      result should equal (x +- 0.1)
      i += 1
    }
  }
}
