import org.scalatest.{FlatSpec, Matchers, ParallelTestExecution}
import tags.{CoverageTest, NonCoverageTest}
import neuralNet.{NeuralNet}
import neuralNet.NeuralNetUtilities.neuralNetFeatureVectorForStateAction

class NeuralNetSpec extends FlatSpec with Matchers with ParallelTestExecution {
  "A NeuralNet" should "correctly convert a state and action into a featureVector" taggedAs(CoverageTest) in {
    var featureVetor = neuralNetFeatureVectorForStateAction(List("X", "", "", "", "", "", "" , "", ""))
    featureVetor should equal (Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    featureVetor = neuralNetFeatureVectorForStateAction(List("X", "", "", "O", "", "O", "" , "", ""))
    featureVetor should equal (Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0))
  }

  it should "be able to learn sin(x)" taggedAs(CoverageTest) in {
    val neuralNet = new NeuralNet(1, 20, 0.05, 1.0)
    var i = 0
    while (i < 100000) { // Train
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      neuralNet.train(Array(x), y)
      i += 1
    }
    i = 0
    while (i < 1000) {
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      val result = neuralNet.feedForward(Array(x))
      result should equal (y +- 0.1)
      i += 1
    }
    while (i < 1000) { // Negative test to check that the test itself isn't broken
      val x = scala.util.Random.nextDouble()
      val y = scala.math.sin(x)
      val result = neuralNet.feedForward(Array(x))
      result should not equal (y +- 0.01)
      i += 1
    }
  }

  it should "be able to learn x=y" taggedAs(CoverageTest) in {
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
      result should equal (x +- 0.1)
      i += 1
    }
  }
}
