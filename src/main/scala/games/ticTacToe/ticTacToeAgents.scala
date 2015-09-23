// Standard Library
import scala.collection.mutable
import scala.util.Random.nextInt
// Custom
import ticTacToeEnvironment.{Constants, Parameters, EnvironmentUtilities}
import neuralNet.{MultiLayerPerceptron}
import activationFunctions.{LinearActivationFunction, TangentSigmoidActivationFunction}
import debug.DebugUtilities.{debugPrint}
import learningFunctions.{UpdateFunctionTypes}

package ticTacToeAgents {

/** TicTacToe Agent types */
object TicTacToeAgentTypes extends Enumeration {
  type TicTacToeAgentType = Value
  val Random, Tabular, Neural = Value
}

object ConvenienceConstructor {
  def makeAgent(_name : String, _type : TicTacToeAgentTypes.TicTacToeAgentType) : TicTacToeAgent = _type match {
    case TicTacToeAgentTypes.Random => new TicTacToeAgentRandom(_name)
    case TicTacToeAgentTypes.Tabular => new TicTacToeAgentTabular(_name)
    case TicTacToeAgentTypes.Neural => new TicTacToeAgentNeural(_name)
  }
}

class TicTacToeAgentNeural(_name : String) extends TicTacToeAgent(_name) {
  val neuralNets = {
    for {i <- Vector.range(0, Constants.boardSize)}
      yield new MultiLayerPerceptron(Vector(Constants.featureVectorSize, Parameters.neuralNumberHiddenNeurons, 1), Vector(new TangentSigmoidActivationFunction(), new LinearActivationFunction()))
  }

  /** Use a neural network to choose the greedy action to take */
  def chooseGreedyAction(boardState : Vector[String]) : Int = {
    maxNeuralNetValueAndActionForState(boardState)._2
  }

  def neuralNetFeatureVectorForStateAction(state : Vector[String]) : Vector[Double] = {
    def iterateState(state : Vector[String], xList : Vector[Double], oList : Vector[Double]) : (Vector[Double], Vector[Double]) = state match {
      case head +: tail => head match {
          case "X" => iterateState(state.tail, xList :+ 1.0, oList :+ 0.0)
          case "O" => iterateState(state.tail, xList :+ 0.0, oList :+ 1.0)
          case ""  => iterateState(state.tail, xList :+ 0.0, oList :+ 0.0)
      }
      case _ => (xList, oList) // If it's empty, return the result
    }

    val featureVectors = iterateState(state, Vector(), Vector())
    featureVectors._1 ++ featureVectors._2
  }

  /** Query the neural network for the maximum value for the given board state.  The return tuple is the (maximumValue, correspondingAction) */
  def maxNeuralNetValueAndActionForState(state : Vector[String]) : (Double, Int) = {
    val possibleMoves = EnvironmentUtilities.emptySpaces(state)
    if (possibleMoves.size == 0) { // The value of an end state position is always 0, and there is no position to take next
      (0.0, 0)
    }
    debugPrint(s"${name} is getting max neural net values for spaces ${possibleMoves.mkString(", ")}")
    val stateValues : Map[Int, Double]= ({
      for {
        possibleMove <- possibleMoves
        value = neuralNets(possibleMove).feedForward(neuralNetFeatureVectorForStateAction(state))(0)
      }
        yield (possibleMove, value)
    })(collection.breakOut)
    debugPrint(s"Player is choosing state values from ${stateValues}")
    val maxStateValues = stateValues.groupBy(_._2).maxBy(_._1)._2
    if (maxStateValues.size > 1) {
      debugPrint(s"Have max value state ties on states ${maxStateValues.mkString(", ")}")
    }
    val maxState = maxStateValues.keySet.toVector(scala.util.Random.nextInt(maxStateValues.size)) // Break ties randomly
    val maxValue = maxStateValues(maxState)
    (maxValue, maxState)
  }

  /** Update the value function for a neural network function approximation learner. */
  def applyReward(reward : Double) : Unit = {
    debugPrint(s"Updating ${name}'s neural net for making the move ${a} from the state ${s}")
    val previousStateFeatureVector = neuralNetFeatureVectorForStateAction(s)
    val previousStateActionValue = neuralNets(a).feedForward(previousStateFeatureVector)(0)
    var updateValue = 0.0
    var targetValue = 0.0
    Parameters.updateFunction match {
      case UpdateFunctionTypes.SARSA => {
        val stateFeatureVector = neuralNetFeatureVectorForStateAction(sp1)
        if (!EnvironmentUtilities.isEndState(sp1)) {
          targetValue = neuralNets(ap1).feedForward(stateFeatureVector)(0)
        }
        updateValue = previousStateActionValue + Parameters.neuralValueLearningAlpha * (reward + Parameters.gamma * targetValue - previousStateActionValue)
        debugPrint(s"previousStateActionValue = ${previousStateActionValue} alpha = ${Parameters.neuralValueLearningAlpha} reward = ${reward} gamma = ${Parameters.gamma} stateActionValue = ${targetValue} updateValue = ${updateValue}")
      }
      case UpdateFunctionTypes.QLearning => {
        if (!EnvironmentUtilities.isEndState(sp1)) {
          targetValue = maxNeuralNetValueAndActionForState(sp1)._1
        }
        updateValue = previousStateActionValue + Parameters.neuralValueLearningAlpha * (reward + Parameters.gamma * targetValue - previousStateActionValue)
      }
    }
    neuralNets(a).train(previousStateFeatureVector, Vector(updateValue))
    debugPrint(s"Updated player ${name}'s neural net for ${previousStateFeatureVector.mkString(", ")} with reward ${reward} and targetValue ${updateValue}")
    val previousStateActionValueUpdated = neuralNets(a).feedForward(previousStateFeatureVector)(0)
    sanityCheckValueUpdate(reward, previousStateActionValue, previousStateActionValueUpdated, targetValue)
  }
}

class TicTacToeAgentTabular(_name : String) extends TicTacToeAgent(_name) {
  type stateValuesType = mutable.HashMap[Int, Double]
  val stateValues = mutable.HashMap[Vector[String], stateValuesType]()  // The state-value function is stored in a map with keys that are environment states of the Tic-tac-toe board and values that are arrays of the value of each possible action in this state.  A possible action is any space that is not currently occupied.

  /** Convenience method for initializing values for a given state if not already initialized */
  def getStateValues(state : Vector[String]) : stateValuesType = {
    if (!stateValues.contains(state)) { // Initialize the state values to 0
      if (EnvironmentUtilities.isEndState(state)) {  // The state values in the stop state are always 0, so always return a map full of zeros
        val zeroMap : stateValuesType = ({
          for {i <- 1 until 10}
            yield (i, 0.0)
        })(collection.breakOut)
        stateValues += state -> zeroMap
      }
      else {
        val newStateValues : stateValuesType = (for {emptySpace <- EnvironmentUtilities.emptySpaces(state)}
          yield (emptySpace, 0.0))(collection.breakOut)
        stateValues(state) = newStateValues
        }
    }
    stateValues(state)
  }

  /** Decide what to do given the current environment and return that action. */
  def chooseGreedyAction(boardState : Vector[String]) : Int = {
    val stateValues = getStateValues(boardState)
    val maxValue = stateValues.maxBy(_._2)._2
    val maxValueSpaces = mutable.ArrayBuffer[Int]()
    for ((key, value) <- stateValues) {
      if (value == maxValue) {
        maxValueSpaces += key
      }
    }
    maxValueSpaces(nextInt(maxValueSpaces.size))
  }

  /** Update the value function for a tabular learner. */
  def applyReward(reward : Double) : Unit = {
    // Make sure they're initialized
    getStateValues(s)
    getStateValues(sp1)
    val previousStateValue = getStateValues(s)(a)
    var updateValue = 0.0
    var targetValue = 0.0
    Parameters.updateFunction match {
      case UpdateFunctionTypes.SARSA => {
        if (!EnvironmentUtilities.isEndState(sp1)) {
          targetValue = getStateValues(sp1)(ap1)
        }
        updateValue = (Parameters.tabularAlpha)*(reward + Parameters.gamma * targetValue - getStateValues(s)(a))
      }
      case UpdateFunctionTypes.QLearning => {
        debugPrint(s"alpha = ${Parameters.tabularAlpha} reward = ${reward} maxStateValue = ${getStateValues(sp1).maxBy(_._2)._2} previousStateValue = ${getStateValues(s)(a)}")
        if (!EnvironmentUtilities.isEndState(sp1)) {
          targetValue = getStateValues(sp1).maxBy(_._2)._2
        }
        updateValue = (Parameters.tabularAlpha)*(reward + Parameters.gamma * targetValue - getStateValues(s)(a))
      }
    }
    stateValues(s)(a) += updateValue
    sanityCheckValueUpdate(reward, previousStateValue, stateValues(s)(a), targetValue)
  }
}

class TicTacToeAgentRandom(_name : String) extends TicTacToeAgent(_name) {
  override def chooseAction(epsilon : Double) : Unit = {
    val prospectiveSpaces = EnvironmentUtilities.emptySpaces(sp1)
    ap1 = chooseGreedyAction(sp1)
  }

  def chooseGreedyAction(boardState : Vector[String]) : Int = {
    val prospectiveSpaces = EnvironmentUtilities.emptySpaces(sp1)
    prospectiveSpaces(nextInt(prospectiveSpaces.size))
  }

  def applyReward(reward : Double) : Unit = {}

  override def reward(reward : Double) : Unit = {}
}

/** The agent object who makes decisions on where to places X's and O's.  Because there are two players, players are identified by an integer value. */
abstract class TicTacToeAgent(_name : String) {
  val name = _name
  private var _sp1 : Vector[String] = Vector.fill(Constants.boardSize){""}
  var s = Vector.fill(Constants.boardSize){""}
  def sp1 : Vector[String] = _sp1
  def sp1_=(newState : Vector[String]) : Unit = {
    s = _sp1
    _sp1 = newState
  }
  var a = 0
  private var _ap1 = 0
  def ap1 : Int = _ap1
  def ap1_=(newAction : Int) : Unit = {
    a = _ap1
    _ap1 = newAction
  }
  var movedOnce = false // To know not to update the value function before its first action
  var numberRewards = 0 // Used to sanity check the number of times each agent is rewarded each episode

  def chooseGreedyAction(boardState : Vector[String]) : Int

  /** The agent chooses the next action to take. */
  def chooseAction(epsilon : Double) : Unit = {
    require(epsilon < 0.0 || epsilon >= 1.0, s"epsilon = ${epsilon} was passed in, but it only makes sense if it's greater than 0.0 and less than 1.0.  For a completely random agent, use TicTacToeAgentRandom.")
    if (EnvironmentUtilities.emptySpaces(sp1).size == 0) {  // If the agent is in a stop state, simply update the action that will be used to give reward
      a = ap1
    }
    else {
      val randomHundred = nextInt(100)
      if (randomHundred <= (100 - (epsilon * 100.0) - 1)) { // Exploit: Choose the greedy action and break ties randomly
          ap1 = chooseGreedyAction(sp1)
      }
      else { // Explore: Randomly choose an action
        val prospectiveSpaces = EnvironmentUtilities.emptySpaces(sp1)
        ap1 = prospectiveSpaces(nextInt(prospectiveSpaces.size))
      }
      if (a == 0) { // Actions are always applied from a, so if a is not set to anything, then the "next action" ap1 is not actually known yet, and we set the "previous action" to the same value.
        a = ap1
      }
    }
  }

  def sanityCheckReward(reward : Double) : Unit = {
      require(a != 0, s"An attempt was made to give reward to ${name} while its previous action is ${a}. A player must move at least once to be rewarded for it.")
      assert(EnvironmentUtilities.emptySpaces(s).contains(a), s"${name} is being rewarded for (s, a) (${s}, ${a}), but it isn't possible to take that action in that given state.")
      assert(EnvironmentUtilities.isEndState(sp1) || EnvironmentUtilities.emptySpaces(sp1).contains(ap1), s"${name} is being rewarded for (sp1, ap1) (${sp1}, ${ap1}), but it isn't possible to take that action in that given state.")
    val previousAndCurrentStateDifferences = EnvironmentUtilities.differenceBetweenBoards(s, sp1)
    if (!EnvironmentUtilities.isFullBoard(sp1)) { // Check that the state is paired with an action that's possible (It's not possible to take an action that's already occupied)
      if (previousAndCurrentStateDifferences.size != 2 || !previousAndCurrentStateDifferences.contains("X") || !previousAndCurrentStateDifferences.contains("O")) {
        if (reward == 0.0) {
          assert(false, s"${name} is being given reward ${reward} for moving from ${s} ${sp1}")
        }
      }
    }
  }

  /** On any value update, the value should approach the expected return of the state action pair, not the immediate reward. The target is the reward plus the value of the next state. */
  def sanityCheckValueUpdate(reward : Double, previousValue : Double, updatedValue : Double, targetValue : Double) : Unit = {
    val target = reward + targetValue
    // The difference between the target and the previous value should always be greater than the difference between the target and updated value
    assert(Math.abs(previousValue - target) < Math.abs(updatedValue - target), s"Player ${name} received a reward of ${reward} and the state value was updated from ${previousValue} to ${updatedValue}.  However, it was expected that the value would get closer to the reward ${reward} + target value ${targetValue}")
  }

  /** Apply the given reward, must be implemented by its subclass */
  def applyReward(reward : Double) : Unit

  /** Check that the agent has been rewared the correct number of times */
  def sanityCheckNumberRewards() : Unit = {
    val numberSpacesOccupied = EnvironmentUtilities.spacesOccupiedByAgent(this, sp1).size
    assert(numberRewards == numberSpacesOccupied, s"The agent ${name} was rewarded ${numberRewards} times during this episode.  However, it made ${numberSpacesOccupied} moves in the ending board state ${sp1}.")
  }

  /** The environment calls this to reward the agent for its action. */
  def reward(reward : Double) : Unit = {
    if (movedOnce) { // There's no need to update if the agent is a random player or if the agent hasn't moved yet.
      sanityCheckReward(reward)
      sanityCheckNumberRewards()
      debugPrint(s"Give reward ${reward} to ${name} moving from ${s} with action ${a} to ${sp1}")
      numberRewards += 1
      applyReward(reward)
    }
  }
}

}
