import scala.collection.mutable.ArrayBuffer
import scala.util.Random.nextDouble
import scala.math.pow
import activationFunctions.ActivationFunction

package neuralNet {

/** A multi-layer perceptron defined by the layout [number of input neuron, number of hidden neurons, number of outpout neurons].  The layout must have at least an input, output and single hidden layer.  All extra numbers are used as additional hidden layers. */
class MultiLayerPerceptron(layout : Vector[Int], activationFunctions : Vector[ActivationFunction]) {
  require(layout.size >= 3, "There must be input layer, output layer, and hidden layer sizes specified at the very least.")
  require(layout.size - 1 == activationFunctions.size, s"You must define an activation function for all but the input layer.  You defined a layout of ${layout}, but gave only ${activationFunctions.size} activation functions: ${activationFunctions}")

  def feedForward(inputs : Vector[Double]) : Vector[Double] = {
    Vector()
  }

  def train(inputs : Vector[Double], expectedOutputs : Vector[Double]) : Unit = {

  }
}

}
