// Learn an agent to play a game of Tic-tac-toe using reinforcement learning with an approximated value function.  

// Convention: The Tic-tac-toe board with size 9 will have its spaces numbered 1 through 9 starting in the top left corner moving right along the row and continuing in the leftmost space on the row below.  

// TODO: Implement SARSA
// TODO: Implement agent vs. agent play, which should learn to always tie.

import java.awt.Graphics
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import javax.swing._
import scala.math
import scala.util.Random._
import scala.collection.mutable._


object TicTacToeLearning {
  /** Executed to initiate playing Tic-tac-toe with Q-Learning. */
  def main(args: Array[String]) {
    val frame = new JFrame("Tic Tac Toe")
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.setSize(180, 180)

    val ticTacToeWorldTabular = new TicTacToeWorld(true)
    val ticTacToeWorldNeuralNet = new TicTacToeWorld(false)
    val worlds = Array(ticTacToeWorldTabular, ticTacToeWorldNeuralNet)
    for (ticTacToeWorld <- worlds) {
      var trainSteps = 100000
      var testSteps = 100000
      if (ticTacToeWorld.tabular == true) {
        println("=== Tabular Q Learning:")
      }
      else {
        trainSteps = 200000
        testSteps = 100000
        println("=== Neural Network Q Learning:")
      }
      frame.setContentPane(ticTacToeWorld.ticTacToePanel)
      frame.setVisible(true)
      val agent = ticTacToeWorld.agent
      val environment = ticTacToeWorld.environment

      println(s"Training ${trainSteps} games against a random player.")
      while (environment.totalGames < trainSteps) { // Train for ${trainSteps} games
        iterateGameStep(ticTacToeWorld, 10.0, frame)
      }
      environment.resetGameStats()
      println(s"Testing the trained Q-Learner against ${testSteps} games.  Exploration is disabled.")
      while (environment.totalGames < testSteps) {
        iterateGameStep(ticTacToeWorld, 0.0, frame)
      }
      println(s"The Q-Learner won ${environment.xWins / environment.totalGames * 100}% of ${testSteps} test games against a random player.")
      println(s"The random player won ${environment.oWins} of the ${testSteps} test games.")
      println(s"${environment.stalemates} of the ${testSteps} test games were stalemates.")
      println("")
    }

    System.exit(0)
  }

  /** Take one step in the game: The player takes an action, the other player responds, and the board hands out reward. */
  def iterateGameStep(ticTacToeWorld : TicTacToeWorld, epsilon : Double, frame : JFrame) {
    val agent = ticTacToeWorld.agent
    val environment = ticTacToeWorld.environment
    agent.chooseAction(epsilon)
    environment.applyAction(agent)
    frame.repaint()
    // TODO: Show some text on the tic tac toe board when a certain player wins
    // TODO: Fix the timing such that the end state is visible to the user for a moment.
    //Thread.sleep(500)
  }

}


/** A TicTacToeWorld contains an Agent and an Environment as well as the TicTacToePanel responsible for drawing the two on screen. */
class TicTacToeWorld(_tabular : Boolean) {
  def tabular = _tabular
  val agent = new Agent("X", _tabular)
  val environment = new Environment()
  val ticTacToePanel = new TicTacToePanel(this)
}

object NeuralNetUtilities {
    /** Take a state and represent it in a way that can be fed into the neural net */
  def neuralNetFeatureVectorForStateAction(state : List[String], action : Int) : Array[Double] = {
    val featureVector : ArrayBuffer[Double] = ArrayBuffer()
    for (owner <- state) {
      if (owner == "X") {
        featureVector += 1.0
      }
      else if (owner == "O") {
        featureVector += -1.0
      }
      else {
        featureVector += 0.0
      }
    }
    featureVector += action
    return featureVector.toArray
  }
}

import NeuralNetUtilities._

/** The agent object who makes decisions on where to places X's and O's.  Because there are two players, players are identified by an integer value.*/
class Agent(_name : String, _tabular : Boolean) {
  val name = _name
  private var _state : List[String] = List.fill(9){""}
  var previousState = List.fill(9){""}
  def state = _state
  def state_=(newState : List[String]): Unit = {
    previousState = _state
    _state = newState
  }
  var newlyOccupiedSpace = 0
  val stateValues = Map[List[String], Map[Int, Double]]()  // The state-value function is stored in a map with keys that are environment states of the Tic-tac-toe board and values that are arrays of the value of each possible action in this state.  A possible action is any space that is not currently occupied.  
  def tabular = _tabular
  var neuralNet : Option[NeuralNet] = None
  if (tabular == false) {
    neuralNet = Option(new NeuralNet(10, 26))
  }

  /** Convenience method for initializing values for a given state if not already initialized */
  def getStateValues(state : List[String]) : Map[Int, Double] = { 
    if (stateValues.contains(state) == false) { // Initialize the state values to 0
      if (EnvironmentUtilities.isFullBoard(state) == true) {  // The state values in the stop state are always 0, so always return a map full of zeros
        val zeroMap = Map[Int, Double]()
        for (i <- 1 until 10) {
          zeroMap(i) = 0.0
        }
        stateValues(state) = zeroMap
      }
      else {
        val emptySpaces = EnvironmentUtilities.emptySpaces(state)
        val newStateValues = Map[Int, Double]()
        for (emptySpace <- emptySpaces) {
          // If taking this space would result in a win, then set to 1.0
          // If taking this space would result in a loss or stalemate, then set to 
          newStateValues(emptySpace) = 0.0
        }
        stateValues(state) = newStateValues
      }
    }
    return stateValues(state)
  }

  /** Query the neural network for the maximum value for the given board state.  The return tuple is the (maximumValue, correspondingAction) */
  def maxNeuralNetValueAndActionForState(state : List[String]) : (Double, Int) = {
    val possibleMoves = EnvironmentUtilities.emptySpaces(state)
    var maxValue = 0.0
    var greedyAction  = 0
    for (possibleMove <- possibleMoves) {
      val input = neuralNetFeatureVectorForStateAction(state, possibleMove)
      val value = neuralNet.get.feedForward(input.toArray)
      if (value > maxValue) {
        greedyAction = possibleMove
        maxValue = value
      }
    }
    return (maxValue, greedyAction)
  }

  /** The agent chooses the next action to take. */
  def chooseAction(exploreEpsilon : Double) {
    val randomHundred = nextInt(100)
    if (randomHundred <= (100 - exploreEpsilon - 1)) { // Exploit: Choose the greedy action and break ties randomly
      if (tabular == true) {
        newlyOccupiedSpace = tabularGreedyAction()
      }
      else {
        newlyOccupiedSpace = neuralNetGreedyAction()
      }
    }
    else { // Explore: Randomly choose an action
      val emptySpaces = EnvironmentUtilities.emptySpaces(state)
      newlyOccupiedSpace = emptySpaces(nextInt(emptySpaces.size))
    }
  }

  /** Use a neural network to choose the greedy action to take */
  def neuralNetGreedyAction() : Int = {
    return maxNeuralNetValueAndActionForState(state)._2
  }

  /** Decide what to do given the current environment and return that action. */
  def tabularGreedyAction() : Int = {
    val stateValues = getStateValues(state)
    val maxValue = stateValues.maxBy(_._2)._2
    val maxValueSpaces = ArrayBuffer[Int]()
    for ((key, value) <- stateValues) {
      if (value == maxValue) {
        maxValueSpaces += key
      }
    }
    return maxValueSpaces(nextInt(maxValueSpaces.size))
  }

  /** The environment calls this to reward the agent for its action. */
  def reward(reward : Double) {
    if (tabular) {
      // Make sure they're initialized
      getStateValues(previousState)
      getStateValues(state)
      val updateValue = (0.70)*((reward + stateValues(state).maxBy(_._2)._2) - stateValues(previousState)(newlyOccupiedSpace)) // Q-Learning
      stateValues(previousState)(newlyOccupiedSpace) += updateValue
    }
    else {
      val previousStateFeatureVector = neuralNetFeatureVectorForStateAction(previousState, newlyOccupiedSpace)
      val previousStateValue = neuralNet.get.feedForward(previousStateFeatureVector)
      val stateMaxValue = maxNeuralNetValueAndActionForState(state)._1
      val discountRate = 0.2
      val learningRate = 0.2 
      val targetValue = previousStateValue + learningRate * (reward + discountRate * stateMaxValue - previousStateValue)  // q(s,a) + learningrate * (reward + discountRate * q'(s,a) - q(s,a))
      neuralNet.get.train(previousStateFeatureVector, targetValue)
    }
  }
}

object EnvironmentUtilities {
  /** Return all spaces where the given agent currently has its mark. */
  def spacesOccupiedByAgent(agent : Agent, spaceOwners : List[String]) : List[Int] = {
    return spaceOwners.zipWithIndex.collect {
      case (element, index) if element == agent.name => index + 1
    }
  }

  /** Return all spaces that have any player on it. */
  def occupiedSpaces(spaceOwners : List[String]) : List[Int] = {
    return spaceOwners.zipWithIndex.collect {
      case (element, index) if element != "" => index + 1
    }
  }

  /** Return a list of all spaces that are not occupied by any player. */
  def emptySpaces(spaceOwners : List[String]) : List[Int] = {
    return spaceOwners.zipWithIndex.collect {
      case (element, index) if element == "" => index + 1
    }
  }

  /** The board is full if all spaces are taken by some agent.  This is used with isWinningBoard() to determine a stalemate. */
  def isFullBoard(spaceOwners : List[String]) : Boolean = {
    val noOwnerIndex = spaceOwners.indexWhere( _ == "")
    if (noOwnerIndex == -1) {
      return true
    }
    else {
      return false
    }
  }
}

/** The environment is responsible for transitioning state and giving reward. */
class Environment() {
  val gridWidth = 30    // Width of each box in the grid
  val size = 3
  var spaceOwners = MutableList.fill(size*size){""}  // Array of each space on the board with the corresponding agent name that is currently occupying the space.  0 if no one is occupying the space.
  
  /** Take a position in the grid and return the left to right number that is the column this position is in. */
  def columnNumber(position : Int) : Int = {
    var column = position % size
    if (column == 0) {
      column = size
    }
    return column - 1
  }

  /** Take a position in the grid and return the top to bottom number that is the row this position is in. */
  def rowNumber(position : Int) : Int = {
    return math.ceil(position.toDouble / size.toDouble).toInt - 1
  }

  // TODO: The board could be represented in a clever way such that looking up the row and column numbers each time is not necessary.
  /** Take a list of all spots owned by a single agent.  This agent is in a winning state if it owns an entire row, an entire column, or an entire diagonal.*/
  def isWinningBoard(spaces : List[Int]) : Boolean = {
    if (spaces.size == 0) {
      return false
    }
    val ownedRows = spaces.map(x => rowNumber(x))
    val ownedColumns = spaces.map(x => columnNumber(x))
    val mostCommonRow = ownedRows.groupBy(identity).mapValues(_.size).maxBy(_._2)
    if (mostCommonRow._2 == size) { // Three spots in the same row, so win
      return true
    }
    val mostCommonColumn = ownedColumns.groupBy(identity).mapValues(_.size).maxBy(_._2)
    if (mostCommonColumn._2 == size) { // Three spots in the same column, so win
      return true
    }
    if ((spaces.contains(1) == true && spaces.contains(5) == true && spaces.contains(9)) == true || (spaces.contains(7) == true && spaces.contains(5) == true && spaces.contains(3) == true)) {
      return true
    }
    else {
      return false
    }
  }

  /** Check if player X won. */
  def xWon() : Boolean = {
    return playerWon("X")
  }

  /** Check if player O won. */
  def oWon() : Boolean = {
    return playerWon("O")
  }

  /** Check if the given player won. */
  def playerWon(playerMark : String) : Boolean = {
    val ownedSpaces = spaceOwners.zipWithIndex.collect {
      case (element, index) if element == playerMark => index + 1
    }
    if (isWinningBoard(ownedSpaces.toList) == true) {
      return true
    }
    return false
  }

  /** Check if the current board state is the end of a game because someone won or it's a tie. */
  def isEndState() : Boolean = {
    if (xWon() == true) {
      return true
    }
    if (oWon() == true) {
      return true
    }
    if (EnvironmentUtilities.isFullBoard(spaceOwners.toList) == true) {
      return true
    }
    return false
  }

  /** Statistics to track the progress of the players through episodes. */
  var xWins = 0.0
  var oWins = 0.0
  var stalemates = 0.0
  var totalGames = 0.0

  /** Clear the statistics */
  def resetGameStats() {
    xWins = 0.0
    oWins = 0.0
    stalemates = 0.0
    totalGames = 0.0
  }

  /** Reset the agent and states for a new episode */
  def endEpisode(agent : Agent) {
    spaceOwners = MutableList.fill(size*size){""}
    agent.previousState = List.fill(size*size){""}
    agent.state = List.fill(size*size){""}
  }

  /** Make the action most recently chosen by the agent take effect. */
  def applyAction(agent : Agent) {
    spaceOwners(agent.newlyOccupiedSpace - 1) = "X" // Take the space chosen by X
    if (isEndState() == true) { // X's move just pushed it into either a winning state or a stalemate
      giveReward(agent)  // newState = old + X's action
    }
    else { // If the game is not over, fill a space randomly with O and give reward. 
      val emptySpaces = EnvironmentUtilities.emptySpaces(spaceOwners.toList)
      val randomSpace = emptySpaces(nextInt(emptySpaces.size))
      spaceOwners(randomSpace - 1) = "O"
      giveReward(agent)  // newState = old + X's action + O action
    }
  }

  /** Update the agent's state and give it a reward for its ation. Return 1 if this is the end of the episode, 0 otherwise. */
  def giveReward(agent : Agent) {
    agent.state = spaceOwners.toList
    if (xWon() == true) {
      agent.reward(1.0)
      xWins += 1.0
    }
    else if (oWon() == true) {
      agent.reward(0.0)
      oWins += 1.0
    }
    else if (EnvironmentUtilities.isFullBoard(spaceOwners.toList) == true) {
      agent.reward(0.5)
      stalemates += 1.0
    }
    else {
      agent.reward(0.0)
    }
    if (isEndState() == true) {
      totalGames += 1.0
      endEpisode(agent)
    }
  }
}


/** The 2D panel that's visible.  This class is responsible for all drawing. */
class TicTacToePanel(gridWorld : TicTacToeWorld) extends JPanel {
  val worldOffset = 40  // So it's not in the very top left corner

  def drawTicTacToeWorld(graphics : Graphics) {
    drawEnvironment(gridWorld.environment, graphics)
    drawSpaceOwnership(gridWorld.environment, graphics)
  }

  /** Logic for drawing the grid on the screen. */
  def drawEnvironment(environment : Environment, graphics : Graphics) {
    val n = environment.size
    for (b <- 1 to n) { // Draw each row
      var y = (b - 1) * environment.gridWidth + worldOffset
      for (a <- 1 to n) { // Draw a single row
        var x = (a - 1) * environment.gridWidth + worldOffset
        graphics.drawRect(x, y, environment.gridWidth, environment.gridWidth)
      }
    }
  }

  /** Logic for drawing the agent's state on the screen. */
  def drawSpaceOwnership(environment : Environment, graphics : Graphics) {
    val n = environment.size
    val circleDiameter = 20
    val rectangleOffset = (worldOffset/2 - circleDiameter/2)/2 // Put it in the center of the grid's box
    var i = 1
    for (owner <- environment.spaceOwners) {
      if (owner != "") {
        val py = environment.rowNumber(i)
        val px = environment.columnNumber(i)
        // TODO: Center these perfectly
        val x = worldOffset + px*environment.gridWidth + environment.gridWidth/4
        val y = worldOffset + py*environment.gridWidth + 23
        val font = new Font("Dialog", Font.PLAIN, 22)
        graphics.setFont(font)
        graphics.asInstanceOf[Graphics2D].setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)  // Enable anti-aliasing
        graphics.drawString(owner, x, y)
      }
      i += 1
    }
  }

  /** Called every time the frame is repainted. */
  override def paintComponent(graphics : Graphics) {
    drawTicTacToeWorld(graphics)
  }
}

TicTacToeLearning.main(args)

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

def testFunction(input : Double) : Double = {
  val value = Math.sin(input)
  return value.toDouble
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

  def train(input : Array[Double], actual : Double) : Double = {
    val result = feedForward(input)
    val error = actual - result
    val deltaOutput = result * (1 - result) * error
    backpropogate(deltaOutput)
    return result
  }

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

