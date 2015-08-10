import scala.collection.mutable._
import scala.util.Random._

package neuralNet {

  /** A simple neural network with a single input neuron and a single output neuron and a given number of hidden neurons. */
  class NeuralNet(numberInputNeurons : Int , numberHiddenNeurons : Int) {
    private val _outputNeuron = new Neuron(false)
    private val _hiddenNeurons : ArrayBuffer[Neuron] = ArrayBuffer()
    private val _inputNeurons : ArrayBuffer[Neuron] = ArrayBuffer()
    private val learningRate = 0.9

    for (i <- 0 until numberInputNeurons) { // Create input neurons
      _inputNeurons += new Neuron(false)
    }

    for (i <- 0 until numberHiddenNeurons) { // Create hidden neurons
      _hiddenNeurons += new Neuron(false)
    }

    // Create bias neurons
    _inputNeurons += new Neuron(true)
    _hiddenNeurons += new Neuron(true)

    for (inputNeuron <- _inputNeurons) {
      for (hiddenNeuron <- _hiddenNeurons) {
        new Connection(inputNeuron, hiddenNeuron)   // Connect the inputs to the hidden neurons
      }
    }

    for (hiddenNeuron <- _hiddenNeurons) {
      new Connection(hiddenNeuron, _outputNeuron) // Connect the hidden neuron to the output neuron
    }

    /** Take a supervised output value and backpropogate the error through the neural net. */
    def train(input : Array[Double], actual : Double) : Double = {
      val result = feedForward(input)
      val error = actual - result
      val deltaOutput = result * (1 - result) * error
      backpropogate(deltaOutput)
      return result
    }

    /** Initiate backpropogation. */
    def backpropogate(deltaOutput : Double) {
      updateOutputWeight(deltaOutput)
      updateHiddenWeights(deltaOutput)
    }

    /** Update the weights of connections to the output neuron */
    def updateOutputWeight(deltaOutput : Double) {
      for (connection <- _outputNeuron.connections) {
        val neuron = connection.a
        val neuronOutput = neuron.output
        val deltaWeight = neuronOutput * deltaOutput
        connection.adjustWeight(learningRate * deltaWeight)
      }
    }

    /** Update the weights of the connections leading to the hidden layer */
    def updateHiddenWeights(deltaOutput : Double) {
      for (hiddenNeuron <- _hiddenNeurons) { // Update each hidden neuron's input connection weight
        if (hiddenNeuron.isBiasNeuron == false) {
          var outputConnectionSum = 0.0
          for (connection <- hiddenNeuron.connections) {
            if (connection.a == hiddenNeuron) { // From hidden layer to output layer
              outputConnectionSum += connection.weight * deltaOutput
            }
          }
          for (connection <- hiddenNeuron.connections) {
            if (connection.b == hiddenNeuron) { // From input layer to hidden layer
              val hiddenNeuronOutput = hiddenNeuron.output
              val hiddenDelta = hiddenNeuronOutput * (1 - hiddenNeuronOutput) * outputConnectionSum
              val deltaWeight = connection.a.output * hiddenDelta
              connection.adjustWeight(learningRate * deltaWeight)
            }
          }
        }
      }
    }

    /** Return the neural net's output value for a given input. */
    def feedForward(inputs : Array[Double]) : Double = {
      var i = 0
      for (input <- inputs) {
        val inputNeuron = _inputNeurons(i)
        inputNeuron.input = input
        inputNeuron.updateOutput() // Make sure it stores the new input value
        i += 1
      }
      for (hiddenNeuron <- _hiddenNeurons) {
        hiddenNeuron.updateOutput()
      }
      _outputNeuron.updateOutput()
      return _outputNeuron.output()
    }
  }


  /** A single neuron in the network, responsible for calculating values */
  class Neuron(_isBiasNeuon : Boolean) {
    private var _sum = 0.0
    var connections : ArrayBuffer[Connection] = ArrayBuffer()
    var input = 0.0
    def isBiasNeuron = _isBiasNeuon
    if (isBiasNeuron) {
      _sum = 0.15 // This is in the range [0, f(n)] where n is the number of input neurons and f(x) = 1/sqrt(n).   See here: http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
    }
    
    /* This is called externally when something has changed so that this neuron's value needs to be updated. */
    def updateOutput() {
      if (isBiasNeuron == false) { // Bias neurons have no calculations to perform
        var sum = 0.0
        var bias = 0.0
        var hasInputConnection = false
        for (connection <- connections) {
          if (connection.b == this) { // This is a connection that inputs into this neuron
            hasInputConnection = true
            val inputNeuron = connection.a
            val weightedValue = inputNeuron.output()*connection.weight
            if (connection.a.isBiasNeuron == true) {
              bias = weightedValue
            }
            else {
              sum += weightedValue
            }
          }
        }
        if (hasInputConnection == true) {
          _sum = sigmoid(sum + bias)
        }
        else { // This is an input neuron
          _sum = input
        }
      }
    }

    /** Sigmoid activation function */
    def sigmoid(input : Double) : Double = {
      return 1.0 / (1.0 + Math.exp(-input))
    }

    /** Return the most recently calculated value */
    def output() : Double = {
      return _sum
    }
  }


  /** A connection is the bond between two neurons.  a is the source of a signal and it's sent to b, modified by the weight of this connection. */
  class Connection(_a : Neuron, _b : Neuron) {
    def a = _a
    def b = _b
    a.connections += this
    b.connections += this
    var weight : Double = nextDouble() * 2 - 1

    def adjustWeight(deltaWeight : Double) {
      weight += deltaWeight
    }
  }

}
