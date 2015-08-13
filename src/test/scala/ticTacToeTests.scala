import org.scalatest._
import collection.mutable.Stack
import neuralNet._
import neuralNet.NeuralNetUtilities._

abstract class UnitSpec extends FlatSpec with Matchers with
  OptionValues with Inside with Inspectors

class ExampleSpec extends FlatSpec with Matchers {

  "A NeuralNet" should "correctly convert a state and action into a featureVector" in {
    var featureVetor = neuralNetFeatureVectorForStateAction(List("X", "", "", "", "", "", "" , "", ""), 2)
    featureVetor should equal (Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    featureVetor = neuralNetFeatureVectorForStateAction(List("X", "", "", "O", "", "O", "" , "", ""), 9)
    featureVetor should equal (Array(1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 9.0))
  }

  it should "throw NoSuchElementException if an empty stack is popped" in {
    val emptyStack = new Stack[Int]
    a [NoSuchElementException] should be thrownBy {
      emptyStack.pop()
    } 
  }
}
