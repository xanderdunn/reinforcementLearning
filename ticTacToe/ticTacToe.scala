// Learn an agent to play a game of Tic-tac-toe using reinforcement learning with an approximated value function.  

// Convention: The Tic-tac-toe board with size 9 will have its spaces numbered 1 through 9 starting in the top left corner moving right along the row and continuing in the leftmost space on the row below.  

// TODO: Implement SARSA
// TODO: Implement SARSA(lambda)
// TODO: Implement agent vs. agent play, which should learn to always tie.
// TODO: Use [breeze](https://github.com/scalanlp/breeze/wiki/Quickstart#breeze-viz) or [wisp](https://github.com/quantifind/wisp) libraries to plot the performance of your learner over time.  x = Number of episodes used to train.  y = Average reward received.  Get this average by running the game 2000 times and store the value at each time step.

// Standard Library
import java.awt.Graphics
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import javax.swing.JFrame
import javax.swing.JPanel
import scala.math
import scala.util.Random._
import scala.collection.mutable._
// Custom
import neuralNet._
import neuralNet.NeuralNetUtilities._
import EnvironmentUtilities._


object TicTacToeLearning {
  /** Executed to initiate playing Tic-tac-toe with Q-Learning. */
  def main(args: Array[String]) {
    val frame = new JFrame("Tic Tac Toe")
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.setSize(180, 180)

    val ticTacToeWorldTabularRandom = new TicTacToeWorld(true, false, true)
    val ticTacToeWorldNeuralNetRandom = new TicTacToeWorld(false, false, true)
    val worlds = Array(ticTacToeWorldTabularRandom, ticTacToeWorldNeuralNetRandom)
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
      val agent = ticTacToeWorld.agent1
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
    val agent = ticTacToeWorld.agent1
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
class TicTacToeWorld(_tabular : Boolean, agent1Random : Boolean, agent2Random : Boolean) {
  def tabular = _tabular
  val agent1 = new Agent("X", _tabular, agent1Random)
  val agent2 = new Agent("O", _tabular, agent2Random)
  val agents = List(agent1, agent2)
  val environment = new Environment(agent1, agent2)
  val ticTacToePanel = new TicTacToePanel(this)
}


/** The agent object who makes decisions on where to places X's and O's.  Because there are two players, players are identified by an integer value.*/
class Agent(_name : String, _tabular : Boolean, _random : Boolean) {
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
  def random = _random

  /** Convenience method for initializing values for a given state if not already initialized */
  def getStateValues(state : List[String]) : Map[Int, Double] = { 
    if (stateValues.contains(state) == false) { // Initialize the state values to 0
      if (isFullBoard(state) == true) {  // The state values in the stop state are always 0, so always return a map full of zeros
        val zeroMap = Map[Int, Double]()
        for (i <- 1 until 10) {
          zeroMap(i) = 0.0
        }
        stateValues(state) = zeroMap
      }
      else {
        val newStateValues = Map[Int, Double]()
        for (emptySpace <- emptySpaces(state)) {
          // TODO: If taking this space would result in a win, then set to 1.0.  If taking this space would result in a loss or stalemate, then set to 0.0.  Otherwise, set to 0.5.
          newStateValues(emptySpace) = 0.0
        }
        stateValues(state) = newStateValues
      }
    }
    return stateValues(state)
  }

  /** Query the neural network for the maximum value for the given board state.  The return tuple is the (maximumValue, correspondingAction) */
  def maxNeuralNetValueAndActionForState(state : List[String]) : (Double, Int) = {
    val possibleMoves = emptySpaces(state)
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
    if (_random) {
      val prospectiveSpaces = emptySpaces(state)
      newlyOccupiedSpace = prospectiveSpaces(nextInt(prospectiveSpaces.size))
    }
    else {
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
        val prospectiveSpaces = emptySpaces(state)
        newlyOccupiedSpace = prospectiveSpaces(nextInt(prospectiveSpaces.size))
      }
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

/** Static convenience functions for handling the environment. */
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

class TicTacToeBoard() {
  private var spaceOwners = emptyMutableList()
  case class CanNotMoveThereException(message: String) extends Exception(message)

  def emptyMutableList() : MutableList[String] = {
    return MutableList.fill(9){""}
  }

  def getList() : List[String] = {
    return spaceOwners.toList
  }

  def setSpaceOwner(space : Int, newOwner : String) {
    val existingOwner = spaceOwners(space - 1)
    if (existingOwner != "") {
      throw new CanNotMoveThereException(s"${newOwner} tried to place someone on space ${space}, but ${existingOwner} is already there.  Board = ${spaceOwners.mkString(", ")}")
    }
    else {
      spaceOwners(space - 1) = newOwner
    }
  }

  def resetBoard() {
    spaceOwners = emptyMutableList()
  }
}

/** The environment is responsible for transitioning state and giving reward. */
class Environment(agent1 : Agent, agent2 : Agent) {
  val size = 3
  var spaceOwners = new TicTacToeBoard()  // Array of each space on the board with the corresponding agent name that is currently occupying the space.  0 if no one is occupying the space.
  
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
  // TODO: Move this to EnvironmentUtilities
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
    val ownedSpaces = spaceOwners.getList().zipWithIndex.collect {
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
    if (isFullBoard(spaceOwners.getList()) == true) {
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
    spaceOwners.resetBoard()
    agent.previousState = List.fill(size*size){""}
    agent.state = List.fill(size*size){""}
  }

  /** Make the action most recently chosen by the agent take effect. */
  def applyAction(agent : Agent) {
    spaceOwners.setSpaceOwner(agent.newlyOccupiedSpace, "X") // Take the space chosen by X
    if (isEndState() == true) { // X's move just pushed it into either a winning state or a stalemate
      giveReward(agent)  // newState = old + X's action
    }
    else { // If the game is not over, fill a space randomly with O and give reward. 
      agent2.state = spaceOwners.getList()
      val agentChosenRandomAction = agent2.chooseAction(0.0)
      spaceOwners.setSpaceOwner(agent2.newlyOccupiedSpace, "O")
      giveReward(agent)  // newState = old + X's action + O action
    }
  }

  /** Determine who won and add it to the statistics */
  def countEndState() {
    totalGames += 1
    if (xWon() == true) {
      xWins += 1
    }
    else if (oWon() == true) {
      oWins += 1
    }
    else if (isFullBoard(spaceOwners.getList())) {
      stalemates += 1
    }
    else {
      println("ERROR: It makes no sense to reach the end state and agent1 didn't win, agent 2 didn't win, and it wasn't a stalemate.")
    }
  }

  /** Update the agent's state and give it a reward for its ation. Return 1 if this is the end of the episode, 0 otherwise. */
  def giveReward(agent : Agent) {
    agent.state = spaceOwners.getList()
    if (xWon() == true) {
      agent.reward(1.0)
    }
    else if (oWon() == true) {
      agent.reward(0.0)
    }
    else if (isFullBoard(spaceOwners.getList()) == true) {
      agent.reward(0.5)
    }
    else {
      agent.reward(0.0)
    }
    if (isEndState() == true) {
      countEndState()
      endEpisode(agent)
    }
  }
}


/** The 2D panel that's visible.  This class is responsible for all drawing. */
class TicTacToePanel(gridWorld : TicTacToeWorld) extends JPanel {
  val worldOffset = 40  // So it's not in the very top left corner
  val gridWidth = 30    // Width of each box in the grid

  def drawTicTacToeWorld(graphics : Graphics) {
    drawEnvironment(gridWorld.environment, graphics)
    drawSpaceOwnership(gridWorld.environment, graphics)
  }

  /** Logic for drawing the grid on the screen. */
  def drawEnvironment(environment : Environment, graphics : Graphics) {
    val n = environment.size
    for (b <- 1 to n) { // Draw each row
      var y = (b - 1) * gridWidth + worldOffset
      for (a <- 1 to n) { // Draw a single row
        var x = (a - 1) * gridWidth + worldOffset
        graphics.drawRect(x, y, gridWidth, gridWidth)
      }
    }
  }

  /** Logic for drawing the agent's state on the screen. */
  def drawSpaceOwnership(environment : Environment, graphics : Graphics) {
    val n = environment.size
    val circleDiameter = 20
    val rectangleOffset = (worldOffset/2 - circleDiameter/2)/2 // Put it in the center of the grid's box
    var i = 1
    for (owner <- environment.spaceOwners.getList()) {
      if (owner != "") {
        val py = environment.rowNumber(i)
        val px = environment.columnNumber(i)
        // TODO: Center these perfectly
        val x = worldOffset + px*gridWidth + gridWidth/4
        val y = worldOffset + py*gridWidth + 23
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


