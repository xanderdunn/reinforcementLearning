// Learn an agent to play a game of Tic-tac-toe using reinforcement learning with an approximated value function.  

// Convention: The Tic-tac-toe board with size n will have its spaces numbered 1 through n*n starting in the top left corner moving right along the row and continuing in the leftmost space on the row below.  Typically, n == 3.

// TODO: Implement both Q-Learning and SARSA
// TODO: Visualze the domain
// TODO: Implement SARSA lambda (reach goal)
// TODO: Initially play the agent against a player who always chooses an action randomly.  Then play against a second agent (agent vs. agent), which should learn to always tie.

// TODO: Implement a tabular value function and Q-Learning on one of the agents to learn how to play well
// TODO: Implement a multi-layer perceptron to approximate the value function

//import java.awt._
import java.awt.Graphics
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import javax.swing._
import scala.math
import scala.util.Random
import scala.collection.mutable._

object TicTacToeLearning {
  def main(args: Array[String]) {
    val frame = new JFrame("Tic Tac Toe")
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.setSize(180, 180)

    val ticTacToeWorld = new TicTacToeWorld()
    frame.setContentPane(ticTacToeWorld.ticTacToePanel)
    frame.setVisible(true)

    val agents = Array(ticTacToeWorld.agent1, ticTacToeWorld.agent2)

    while (true) {
      val randomAction = ticTacToeWorld.agent2.act(ticTacToeWorld.environment)  // O makes a random play
      val episodeEnded = ticTacToeWorld.environment.takeSpace(ticTacToeWorld.agent2, randomAction)
      frame.repaint()
      Thread.sleep(1)
      if (episodeEnded == 1) {
        println("Episode ended on O")
        ticTacToeWorld.environment.giveReward(ticTacToeWorld.agent1, ticTacToeWorld.agent2, ticTacToeWorld.agent1.newlyOccupiedSpace, agents)
        ticTacToeWorld.endEpisode()
      }
      else {
        ticTacToeWorld.agent1.previousState = ticTacToeWorld.environment.spaceOwners.toList
        val learnedAction = ticTacToeWorld.agent1.act(ticTacToeWorld.environment)
        val episodeEnded = ticTacToeWorld.environment.takeSpace(ticTacToeWorld.agent1, ticTacToeWorld.agent1.newlyOccupiedSpace)
        ticTacToeWorld.environment.giveReward(ticTacToeWorld.agent1, ticTacToeWorld.agent2, ticTacToeWorld.agent1.newlyOccupiedSpace, agents)
        if (episodeEnded == 1) {
          frame.repaint()
          Thread.sleep(1)
          ticTacToeWorld.endEpisode()
        }
      }
      frame.repaint()
      Thread.sleep(1)
    }
  }
}


/** A TicTacToeWorld contains an Agent and an Environment as well as the TicTacToePanel responsible for drawing the two on screen. */
class TicTacToeWorld {
  val agent1 = new Agent("X")
  val agent2 = new Agent("O")
  val environment = new Environment()
  val ticTacToePanel = new TicTacToePanel(this)

  def endEpisode() {
    environment.spaceOwners = MutableList.fill(environment.size*environment.size){""}
    agent1.previousState = List()
    agent2.previousState = List()
  }

}


/** The agent object who makes decisions on where to places X's or O'.  Because there are two players, players are identified by an integer value.*/
class Agent(_name : String) {
  val name = _name
  var newlyOccupiedSpace = 0
  var previousState : List[String] = List()
  val stateValues = Map[List[String], Map[Int, Int]]()  // The state-value function is stored in a map with keys that are environment states of the Tic-tac-toe board and values that are arrays of the value of each possible action in this state.  A possible action is any space that is not currently occupied.  

  /** Convenience method for initializing values for a given state if not already initialized */
  def getStateValues(state : List[String], size : Int) : Map[Int, Int] = { 
    if (!stateValues.contains(state)) { // Initialize the state values to 0
      if (EnvironmentUtilities.isFullBoard(state)) {  // The state values in the stop state are always 0, so always return a map full of zeros
        val zeroMap = Map[Int, Int]()
        for (i <- 1 until size + 1) {
          zeroMap(i) = 0
        }
        stateValues(state) = zeroMap
      }
      else {
        val emptySpaces = EnvironmentUtilities.emptySpaces(state)
        val newStateValues = Map[Int, Int]()
        for (emptySpace <- emptySpaces) {
          newStateValues(emptySpace) = 0
        }
        stateValues(state) = newStateValues
        println(s"There are ${stateValues.size} stateValues")
      }
    }
    return stateValues(state)
  }

  /** Decide what to do given the current environment and return that action. */
  def act(environment : Environment) : Int = {
    if (name == "O") {
      val rand = scala.util.Random
      val emptySpaces = EnvironmentUtilities.emptySpaces(environment.spaceOwners.toList)
      return emptySpaces(rand.nextInt(emptySpaces.size))
    }
    if (previousState.size == 0) {
      previousState = environment.spaceOwners.toList
    }
    val rand = scala.util.Random
    val randomHundred = rand.nextInt(100)
    if (randomHundred <= 89) { // Exploit: Choose the greedy action and break ties randomly
      val stateValues = getStateValues(environment.spaceOwners.toList, environment.size)
      val maxValue = stateValues.max._2
      val maxValueSpaces = ArrayBuffer[Int]()
      for ((key, value) <- stateValues) {
        if (value == maxValue) {
          maxValueSpaces += key
        }
      }
      newlyOccupiedSpace = maxValueSpaces(rand.nextInt(maxValueSpaces.size))
    }
    else { // Explore: Randomly choose an action
      val emptySpaces = EnvironmentUtilities.emptySpaces(environment.spaceOwners.toList)
      newlyOccupiedSpace = emptySpaces(rand.nextInt(emptySpaces.size))
    }
    return newlyOccupiedSpace
  }

  /** The environment calls this to reward the agent for its action. */
  def reward(value : Int, state : List[String], size : Int) {
    if (name == "X") {
      // Make sure they're initialized
      getStateValues(previousState, size)
      getStateValues(state, size)
      //println(s"previousState = ${previousState}")
      //println(s"previousState values = ${getStateValues(previousState, size).mkString(", ")}")
      //println(s"newlyOccupiedSpace = ${newlyOccupiedSpace}")
      //println(s"state = ${state}")
      //println(s"state values = ${getStateValues(state, size).mkString(", ")}")
      //println(s"Got reward ${value}")
      val updateValue = (value + stateValues(state).max._2) - stateValues(previousState)(newlyOccupiedSpace) // Q-Learning
      stateValues(previousState)(newlyOccupiedSpace) += updateValue
      //println(stateValues)
    }
  }
}

object EnvironmentUtilities {
    def spacesOccupiedByAgent(agent : Agent, spaceOwners : List[String]) : List[Int] = {
    return spaceOwners.zipWithIndex.collect {
      case (element, index) if element == agent.name => index + 1
    }
  }

  def occupiedSpaces(spaceOwners : List[String]) : List[Int] = {
    return spaceOwners.zipWithIndex.collect {
      case (element, index) if element != "" => index + 1
    }
  }

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
    // TODO: Generalize the diagonals to n sized boards
    if ((spaces.contains(1) && spaces.contains(5) && spaces.contains(9)) || (spaces.contains(7) && spaces.contains(5) && spaces.contains(3))) {
      return true
    }
    else {
      return false
    }
  }

  def agentWon(candidateAgent : Agent, agents : Array[Agent]) : Boolean = {
    for (agent <- agents) {
      if (isWinningBoard(EnvironmentUtilities.spacesOccupiedByAgent(agent, spaceOwners.toList)) == true && candidateAgent == agent) {
        return true
      }
    }
    return false
  }

  var xWins = 0.0
  var oWins = 0.0
  var stalemates = 0.0
  var totalGames = 0.0

  def takeSpace(agent : Agent, action : Int) : Int = {
    spaceOwners(action - 1) = agent.name
    var episodeEnded = 0
    if (isWinningBoard(EnvironmentUtilities.spacesOccupiedByAgent(agent, spaceOwners.toList))) {
      episodeEnded = 1
    }
    else if (EnvironmentUtilities.isFullBoard(spaceOwners.toList)) {
      episodeEnded = 1
    }
    return episodeEnded
  }

  /** Update the agent's state and give it a reward for its ation. Return 1 if this is the end of the episode, 0 otherwise. */
  def giveReward(agent : Agent, otherAgent : Agent, action : Int, agents : Array[Agent]) {
    println(s"giveReward called with state ${spaceOwners}")
    val oWon = isWinningBoard(EnvironmentUtilities.spacesOccupiedByAgent(otherAgent, spaceOwners.toList))
    if (agentWon(agent, agents) == true) {
      agent.reward(0, spaceOwners.toList, size)
      totalGames += 1.0
      xWins += 1.0
      println(s"${agent.name} WON!")
    }
    else if (oWon == true) {
      agent.reward(-1, spaceOwners.toList, size)
      oWins += 1.0
      totalGames += 1.0
      println(s"${otherAgent.name} WON!")
    }
    else if (EnvironmentUtilities.isFullBoard(spaceOwners.toList) == true) {
      agent.reward(-1, spaceOwners.toList, size)
      println("Stalemate")
      stalemates += 1.0
      totalGames += 1.0
    }
    else {
      agent.reward(-1, spaceOwners.toList, size)
    }
    println(s"X has won ${(xWins/totalGames)*100}% of games")
    println(s"O has won ${(oWins/totalGames)*100}% of games")
    println(s"Stalemate has happened ${(stalemates/totalGames)*100}% of games")
  }
}


/** The 2D panel that's visible.  This class is responsible for all drawing. */
class TicTacToePanel(gridWorld : TicTacToeWorld) extends JPanel {
  val worldOffset = 40  // So it's not in the very top left corner

  def drawTicTacToeWorld(graphics : Graphics) {
    drawEnvironment(gridWorld.environment, graphics)
    drawAgent(gridWorld.agent1, gridWorld.environment, graphics)
    drawAgent(gridWorld.agent2, gridWorld.environment, graphics)
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
  def drawAgent(agent : Agent, environment : Environment, graphics : Graphics) {
    val n = environment.size
    val occupiedSpaces = EnvironmentUtilities.spacesOccupiedByAgent(agent, environment.spaceOwners.toList)
    val circleDiameter = 20
    val rectangleOffset = (worldOffset/2 - circleDiameter/2)/2 // Put it in the center of the grid's box
    for (occupiedSpace <- occupiedSpaces) {
      val py = environment.rowNumber(occupiedSpace)
      val px = environment.columnNumber(occupiedSpace)
      // TODO: Center these perfectly
      val x = worldOffset + px*environment.gridWidth + environment.gridWidth/4
      val y = worldOffset + py*environment.gridWidth + 23
      val font = new Font("Dialog", Font.PLAIN, 22)
      graphics.setFont(font)
      graphics.asInstanceOf[Graphics2D].setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)  // Enable anti-aliasing
      graphics.drawString(agent.name, x, y)
    }
  }

  /** Called every time the frame is repainted. */
  override def paintComponent(graphics : Graphics) {
    drawTicTacToeWorld(graphics)
  }

}

TicTacToeLearning.main(args)

