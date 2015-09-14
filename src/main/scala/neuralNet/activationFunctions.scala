/** Activation functions for the neurons. */

package activationFunctions {
  // All activation functions must implement value() and derviativeValue()
  //def bipolarSigmoidPrime(input: Double) : Double = (1 + sigmoid(input)) * (1 - sigmoid(input)) / 2
  //def bipolarSigmoid(input : Double) : Double = 2/(1 + Math.exp(-input)) - 1
  //def tanhPrime(input : Double) : Double = 3.4318*scala.math.pow((1/scala.math.cosh(2*input)), 2)
  //def tanh(input : Double) : Double = 1.7159 * scala.math.tanh(2/3*input)
  //def oneOverX(input : Double) : Double = input / (1.0 + Math.abs(input))
  //def oneOverXPrime(input : Double) : Double = 1.0 / pow(Math.abs(input) + 1, 2.0)
  
  abstract class ActivationFunction {
    def value(x : Double) : Double
    def derivativeValue(x : Double) : Double
  }

  // TODO: Why does the input for these functions need to be a vector rather than a single x?
  class SigmoidActivationFunction extends ActivationFunction {
  //def sigmoid(input : Double) : Double = 1.0 / (1.0 + Math.exp(-input))
  //def sigmoidPrime(input : Double) : Double = input * (1 - input)
    def value(x : Double) : Double = 1.0

    def derivativeValue(x : Double) : Double = 1.0
  }

  class TangentSigmoidActivationFunction extends ActivationFunction {
    def value(x : Double) : Double = 1.0

    def derivativeValue(x : Double) : Double = 1.0
  }

  class LinearActivationFunction extends ActivationFunction {
  //def identitiy(input : Double) : Double = input
  //def identityPrime(input : Double) : Double = 1.0
    def value(x : Double) : Double = 1.0

    def derivativeValue(x : Double) : Double = 1.0
  }

}
