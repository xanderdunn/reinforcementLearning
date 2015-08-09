// Learn an agent to play a game of Tic-tac-toe using reinforcement learning with an approximated value function.  

// Convention: The Tic-tac-toe board with size n will have its spaces numbered 1 through n*n starting in the top left corner moving right along the row and continuing in the leftmost space on the row below.  Typically, n == 3.

// TODO: Implement both Q-Learning and SARSA
// TODO: Visualze the domain
// TODO: Implement SARSA lambda (reach goal)
// TODO: Implement (agent vs. agent) play, which should learn to always tie.

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

    while (true) {
      val agent = ticTacToeWorld.agent
      val environment = ticTacToeWorld.environment
      val action = agent.chooseAction(environment)
      environment.applyAction(agent)
      frame.repaint()
      Thread.sleep(1)
    }
  }
}


/** A TicTacToeWorld contains an Agent and an Environment as well as the TicTacToePanel responsible for drawing the two on screen. */
class TicTacToeWorld {
  val agent = new Agent("X")
  val environment = new Environment()
  val ticTacToePanel = new TicTacToePanel(this)
}


/** The agent object who makes decisions on where to places X's or O'.  Because there are two players, players are identified by an integer value.*/
class Agent(_name : String) {
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

  /** Convenience method for initializing values for a given state if not already initialized */
  def getStateValues(state : List[String]) : Map[Int, Double] = { 
    if (stateValues.contains(state) == false) { // Initialize the state values to 0
      if (EnvironmentUtilities.isFullBoard(state) == true) {  // The state values in the stop state are always 0, so always return a map full of zeros
        val zeroMap = Map[Int, Double]()
        for (i <- 1 until 4) {
          zeroMap(i) = 0.0
        }
        stateValues(state) = zeroMap
      }
      else {
        val emptySpaces = EnvironmentUtilities.emptySpaces(state)
        val newStateValues = Map[Int, Double]()
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
  def chooseAction(environment : Environment) : Int = {
    val rand = scala.util.Random
    val randomHundred = rand.nextInt(100)
    if (randomHundred <= 89) { // Exploit: Choose the greedy action and break ties randomly
      val stateValues = getStateValues(environment.spaceOwners.toList)
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
  def reward(value : Int) {
      // Make sure they're initialized
      getStateValues(previousState)
      getStateValues(state)
      val updateValue = (0.1)*((value + stateValues(state).max._2) - stateValues(previousState)(newlyOccupiedSpace)) // Q-Learning
      stateValues(previousState)(newlyOccupiedSpace) += updateValue
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

  def xWon() : Boolean = {
    val xSpaces = spaceOwners.zipWithIndex.collect {
      case (element, index) if element == "X" => index + 1
    }
    if (isWinningBoard(xSpaces.toList) == true) {
      return true
    }
    return false
  }

  def oWon() : Boolean = {
    val oSpaces = spaceOwners.zipWithIndex.collect {
      case (element, index) if element == "O" => index + 1
    }
    if (isWinningBoard(oSpaces.toList) == true) {
      return true
    }
    return false
  }

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

  var xWins = 0.0
  var oWins = 0.0
  var stalemates = 0.0
  var totalGames = 0.0

  def endEpisode(agent : Agent) {
    spaceOwners = MutableList.fill(size*size){""}
    agent.previousState = List.fill(size*size){""}
    agent.state = List.fill(size*size){""}
  }

  def applyAction(agent : Agent) {
    spaceOwners(agent.newlyOccupiedSpace - 1) = "X" // Take the space chosen by X
    if (isEndState() == true) { // X's move just pushed it into either a winning state or a stalemate
      giveReward(agent)  // newState = old + X's action
    }
    else { // If the game is not over, fill a space randomly with O and give reward. 
      val emptySpaces = EnvironmentUtilities.emptySpaces(spaceOwners.toList)
      val rand = scala.util.Random
      val randomSpace = emptySpaces(rand.nextInt(emptySpaces.size))
      spaceOwners(randomSpace - 1) = "O"
      giveReward(agent)  // newState = old + X's action + O action
    }
  }

  /** Update the agent's state and give it a reward for its ation. Return 1 if this is the end of the episode, 0 otherwise. */
  def giveReward(agent : Agent) {
    agent.state = spaceOwners.toList
    if (xWon() == true) {
      agent.reward(0)
      xWins += 1.0
      println("X WON!")
    }
    else if (oWon() == true) {
      agent.reward(-1)
      oWins += 1.0
      println("O WON!")
    }
    else if (EnvironmentUtilities.isFullBoard(spaceOwners.toList) == true) {
      agent.reward(-1)
      stalemates += 1.0
      println("Stalemate")
    }
    else {
      agent.reward(-1)
    }
    if (isEndState() == true) {
      totalGames += 1.0
      endEpisode(agent)
      println(s"X has won ${(xWins/totalGames)*100}% of ${totalGames} games")
      println(s"O has won ${(oWins/totalGames)*100}% of games")
      println(s"Stalemate has happened ${(stalemates/totalGames)*100}% of games")
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

