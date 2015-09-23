// Learn an agent to play a game of Tic-tac-toe using reinforcement learning with an approximated value function.

// Convention: The Tic-tac-toe board with size 9 will have its spaces numbered 1 through 9 starting in the top left corner moving right along the row and continuing in the leftmost space on the row below.

// Standard Library
import java.awt.Graphics
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import javax.swing.JFrame
import javax.swing.JPanel
import scala.math
import scala.util.Random.{nextInt, nextDouble}
import scala.collection.mutable
// Custom
import activationFunctions.{TangentSigmoidActivationFunction, LinearActivationFunction}
import ticTacToeEnvironment.EnvironmentUtilities._
import debug.DebugUtilities.debugPrint
import learningFunctions.{UpdateFunctionTypes}
import ticTacToeAgents.{TicTacToeAgent, TicTacToeAgentTypes, ConvenienceConstructor}
import ticTacToeEnvironment.Constants.{X, O}

package ticTacToeEnvironment {

object Constants {
  val boardSize = 9
  val featureVectorSize = 2 * boardSize  // One for each position on the board, both X and O
  val O = "O"
  val X = "X"
}


/** Parameters for the Q value update function and the neural network. */
object Parameters {
  // Tabular Parameters
  val tabularAlpha = 0.1
  // Both
  val epsilon = 0.4
  val gamma = 1.0 // discount rate
  // Neural Net Parameters
  val neuralNetAlpha = 0.5             // The learning rate in the neural net itself
  val neuralInitialBias = 0.33  // This is in the range [0, f(n)] where n is the number of input neurons and f(x) = 1/sqrt(n).   See here: http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
  val neuralNumberHiddenNeurons = 26
  val neuralValueLearningAlpha = 1.0/neuralNumberHiddenNeurons // The learning rate used by the value update function
  val updateFunction = UpdateFunctionTypes.SARSA
}


/** Object to store game-specific parameters.  This is passed to TicTacToeLearning to start a game. */
class GameParameters(_agent1Type : TicTacToeAgentTypes.TicTacToeAgentType, _agent2Type : TicTacToeAgentTypes.TicTacToeAgentType, _numberTrainEpisodes : Int, _numberTestEpisodes : Int) {
  val agent1Type = _agent1Type
  val agent2Type = _agent2Type
  val numberTrainEpisodes = _numberTrainEpisodes
  val numberTestEpisodes = _numberTestEpisodes
}

/** Executed to initiate playing and learning Tic-tac-toe. */
class TicTacToeLearning(generateGraph : Boolean, gameParameters : GameParameters, showVisual : Boolean = false) {
  /** The go button.  Actually initiates game play and learning until the ultimate goal is achieved: Generate a graph or return results in the form (# games X won, # games O won, # stalemates, # total games, # unique board states encountered). */
  def learn() : (Double, Double, Double, Double, Int) = {
    if (generateGraph) { // Set to true if you want to generate graphs instead of initiating single test runs with output in the terminal
      PlotGenerator.generateLearningCurves() // TODO: Use the below code path rather than a separate code path for collecting data to plot on this run.
      System.exit(0)
    }

    val ticTacToeWorld = new TicTacToeWorld(gameParameters.agent1Type, gameParameters.agent2Type)
    val environment = ticTacToeWorld.environment

    // GUI window to visaulize the Tic-tac-toe game
    if (showVisual) {
      val frame = new JFrame("Tic Tac Toe")
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      frame.setSize(180, 180)
      frame.setContentPane(ticTacToeWorld.ticTacToePanel)
      frame.setVisible(true)
    }

    while (environment.totalGames < gameParameters.numberTrainEpisodes) {
      playEpisode(ticTacToeWorld, Parameters.epsilon, true, "")
    }
    environment.resetGameStats()
    while (environment.totalGames < gameParameters.numberTestEpisodes) {
      playEpisode(ticTacToeWorld, 0.0, false, "")
    }
    val uniqueBoardStates = ticTacToeWorld.environment.spaceOwners.uniqueBoardStates
    (environment.xWins, environment.oWins, environment.stalemates, environment.totalGames, uniqueBoardStates.size)
  }

  object PlotGenerator {
    def generateLearningCurves() {
      val settings = Vector((25000, 200, TicTacToeAgentTypes.Tabular, TicTacToeAgentTypes.Random, s"Tabular vs. Random Agent, epsilon=${Parameters.epsilon}  alpha=${Parameters.tabularAlpha}", "tabularVrandom.pdf", 1),
        (50000, 100, TicTacToeAgentTypes.Neural, TicTacToeAgentTypes.Random, s"Neural vs. Random Agent, epsilon=${Parameters.epsilon} learningAlpha=${Parameters.neuralValueLearningAlpha} netAlpha=${Parameters.neuralNetAlpha} gamma=${Parameters.gamma} ${Parameters.neuralNumberHiddenNeurons} hidden neurons ${Parameters.neuralInitialBias} initialBias", "neuralVrandom.pdf", 1),
        (4000, 150, TicTacToeAgentTypes.Tabular, TicTacToeAgentTypes.Tabular, s"Tabular vs. Tabular, epsilon=${Parameters.epsilon}  alpha=${Parameters.tabularAlpha}", "tabularVtabular.pdf", 2),
        (40000, 100, TicTacToeAgentTypes.Neural, TicTacToeAgentTypes.Neural, s"Neural vs. Neural, epsilon=${Parameters.epsilon} learningAlpha=${Parameters.neuralValueLearningAlpha} netAlpha=${Parameters.neuralNetAlpha} gamma=${Parameters.gamma} ${Parameters.neuralNumberHiddenNeurons} hidden neurons ${Parameters.neuralInitialBias} initial bias", "neuralVneural.pdf", 3))

      for (setting <- settings) {
        val numberEpisodes = setting._1
        val numberIterations = setting._2
        val agent1Type = setting._3
        val agent2Type = setting._4
        val title = setting._5
        val filename = setting._6
        val plotting = setting._7 // 1 if I'm plotting player X wins 2 if I'm plotting stalemates and 3 if I'm plotting both player 3 and player 4 wins

        var i = 0
        val episodeNumbers : mutable.Seq[Double] = mutable.Seq.fill(numberEpisodes){0.0}
        while (i < numberEpisodes) {
          episodeNumbers(i) = i.toDouble + 1.0
          i += 1
        }
        var iteration = 0
        val finalResults1 : mutable.Seq[Double] = mutable.Seq.fill(numberEpisodes){0.0}
        val finalResults2 : mutable.Seq[Double] = mutable.Seq.fill(numberEpisodes){0.0}
        while (iteration < numberIterations) {
          println(s"Iteration ${iteration}/${numberIterations}")
          val results = playTrainingSession(numberEpisodes, agent1Type, agent2Type, Parameters.epsilon)
          var i = 0
          for (result <- results) {
            if (plotting == 1) {
              if (result == 1) {
                finalResults1(i) = finalResults1(i) + result
              }
            }
            else if (plotting == 2) {
              if (result == 0) {
                finalResults1(i) = finalResults1(i) + 1
              }
            }
            else if (plotting == 3) {
              if (result == 1) {
                finalResults1(i) = finalResults1(i) + 1
              }
              else if (result == -1) {
                finalResults2(i) = finalResults2(i) + 1
              }
            }
            i += 1
          }
          iteration += 1
        }

        i = 0
        for (result <- finalResults1) {
          finalResults1(i) = finalResults1(i).toDouble / numberIterations.toDouble * 100.0
          i += 1
        }
        i = 0
        for (result <- finalResults2) {
          finalResults2(i) = finalResults2(i).toDouble / numberIterations.toDouble * 100.0
          i += 1
        }
        // TODO: Output the results so that I can graph them externally
      }
    }
  }

  /**  Play a set of episodes for training. */
  // TODO: This can be removed entirely by absorbing it into the learn() method above when the generateGraph boolean is set
  def playTrainingSession(numberEpisodes : Int, agent1Type : TicTacToeAgentTypes.TicTacToeAgentType, agent2Type : TicTacToeAgentTypes.TicTacToeAgentType, epsilon : Double) : Seq[Double] = {
    println(s"Playing a training session with ${numberEpisodes} episodes")
    val stats : mutable.IndexedSeq[Double] = mutable.IndexedSeq.fill(numberEpisodes){0.0}
    var episodeCounter = 0
    val ticTacToeWorld = new TicTacToeWorld(agent1Type, agent2Type)
    while (episodeCounter < numberEpisodes) {
      stats(episodeCounter) = playEpisode(ticTacToeWorld, epsilon, true, X)
      episodeCounter += 1
    }
    stats
  }

  def playEpisode(ticTacToeWorld : TicTacToeWorld, epsilon : Double, shouldLearn : Boolean, collectingDataFor : String) : Double = {
    var episodeOutcome = -2.0
    while (episodeOutcome == -2.0) { // Train with epsilon
      episodeOutcome = iterateGameStep(ticTacToeWorld, epsilon, shouldLearn, None, collectingDataFor)
    }
    episodeOutcome = -2.0
    while (episodeOutcome == -2.0) { // Test run with epsilon = 0
      episodeOutcome = iterateGameStep(ticTacToeWorld, 0.0, shouldLearn, None, collectingDataFor)
    }
    episodeOutcome
  }

  /** Take one step in the game: The player takes an action, the other player responds, and the board hands out reward to both players.  If you're collecting data for a graph, pass in the string "X" or "O" for the player whose data you're interested in.  This method returns 1 if that player won this episode, -1 if it lost, 0 if it was a stalemate, and -2 if the episode hasn't ended. */
  def iterateGameStep(ticTacToeWorld : TicTacToeWorld, epsilon : Double, shouldLearn : Boolean, frame : Option[JFrame], collectingDataFor : String) : Double = {
    val agent = ticTacToeWorld.currentPlayer
    val environment = ticTacToeWorld.environment
    environment.applyAction(agent, epsilon, shouldLearn)
    var returnValue = -2.0
    if (isEndState(environment.spaceOwners.getVector())) {
      if (playerWon(ticTacToeWorld.agent1, environment.spaceOwners.getVector())) {
        returnValue = 1.0
      }
      else if (playerWon(environment.getOtherAgent(ticTacToeWorld.agent1), environment.spaceOwners.getVector())) {
        returnValue = -1.0
      }
      else {
        returnValue = 0.0
      }
      environment.countEndState()
      ticTacToeWorld.endEpisode()
    }
    else {
      ticTacToeWorld.currentPlayer = environment.getOtherAgent(ticTacToeWorld.currentPlayer)
    }
    if (frame != None) {
      frame.get.repaint()
    }
    returnValue
    // TODO: Show some text on the tic tac toe board when a certain player wins
    // TODO: Fix the timing such that the end state is visible to the user for a moment.
    //Thread.sleep(500)
  }
}


/** A TicTacToeWorld contains an Agent and an Environment as well as the TicTacToePanel responsible for drawing the two on screen. */
class TicTacToeWorld(_agent1Type : TicTacToeAgentTypes.TicTacToeAgentType, _agent2Type : TicTacToeAgentTypes.TicTacToeAgentType) {
  val agent1 = ConvenienceConstructor.makeAgent(X, _agent1Type)
  val agent2 = ConvenienceConstructor.makeAgent(O, _agent2Type)
  val agents = Vector(agent1, agent2)
  val environment = new Environment(agent1, agent2)
  val ticTacToePanel = new TicTacToePanel(this)
  var currentPlayer = agent1
  var firstPlayer = agent1
  val xLostStates = scala.collection.mutable.Map[Vector[String], Int]()

  /** Reset the agent and states for a new episode */
  def endEpisode() : Unit = {
    currentPlayer = agents(nextInt(agents.size))
    debugPrint(s"firstPlayer = ${firstPlayer.name}")
    environment.spaceOwners.resetBoard()
    agent1.s = Vector.fill(Constants.boardSize){""}
    agent1.sp1 = Vector.fill(Constants.boardSize){""}
    agent1.movedOnce = false
    agent1.a = 0
    agent1.ap1 = 0
    agent1.numberRewards = 0
    agent2.s = Vector.fill(Constants.boardSize){""}
    agent2.sp1 = Vector.fill(Constants.boardSize){""}
    agent2.movedOnce = false
    agent2.a = 0
    agent2.ap1 = 0
    agent2.numberRewards = 0
  }
}


/** Static convenience functions for handling the environment. */
object EnvironmentUtilities {
  val size = 3
  /** Return all spaces where the given agent currently has its mark. */
  def spacesOccupiedByAgent(agent : TicTacToeAgent, spaceOwners : Vector[String]) : Vector[Int] = {
    spaceOwners.zipWithIndex.collect {
      case (element, index) if element == agent.name => index + 1
    }
  }

  /** Take a position in the grid and return the left to right number that is the column this position is in. */
  def columnNumber(position : Int) : Int = {
    var column = position % size
    if (column == 0) {
      column = size
    }
    column - 1
  }

  /** Take a position in the grid and return the top to bottom number that is the row this position is in. */
  def rowNumber(position : Int) : Int = {
    math.ceil(position.toDouble / size.toDouble).toInt - 1
  }

  // TODO: The board could be represented in a clever way such that looking up the row and column numbers each time is not necessary.
  /** Take a Vector of 0's and 1's where 1 is each spot owned by a single agent.  This agent is in a winning state if it owns an entire row, an entire column, or an entire diagonal. */
  def isWinningBoard(spaces : Vector[Int]) : Boolean = {
    if (spaces.size == 0) {
      false
    }
    val ownedRows = spaces.map(x => rowNumber(x))
    val ownedColumns = spaces.map(x => columnNumber(x))
    val mostCommonRow = ownedRows.groupBy(identity).mapValues(_.size).maxBy(_._2)
    if (mostCommonRow._2 == size) { // Three spots in the same row, so win
      true
    }
    val mostCommonColumn = ownedColumns.groupBy(identity).mapValues(_.size).maxBy(_._2)
    if (mostCommonColumn._2 == size) { // Three spots in the same column, so win
      true
    }
    if ((spaces.contains(1) && spaces.contains(5) && spaces.contains(9)) || (spaces.contains(7) && spaces.contains(5) && spaces.contains(3))) {
      true
    }
    else {
      false
    }
  }

  /** Return all spaces that have any player on it. */
  def occupiedSpaces(spaceOwners : Vector[String]) : Vector[Int] = {
    spaceOwners.zipWithIndex.collect {
      case (element, index) if element != "" => index + 1
    }
  }

  /** Return a list of all spaces that are not occupied by any player. */
  def emptySpaces(spaceOwners : Vector[String]) : Vector[Int] = {
    spaceOwners.zipWithIndex.collect {
      case (element, index) if element == "" => index + 1
    }
  }

  /** Compares two boards and returns the values in boardState2 that are different from boardState1 */
  def differenceBetweenBoards(boardState1 : Vector[String], boardState2 : Vector[String]) : Vector[String] = {
    (for {
      i <- 0 until boardState1.length
      board1Position = boardState1(i)
      board2Position = boardState2(i)
      if (board1Position != board2Position)
    }
      yield board2Position)(collection.breakOut)
  }

  /** The board is full if all spaces are taken by some agent.  This is used with isWinningBoard() to determine a stalemate. */
  def isFullBoard(spaceOwners : Vector[String]) : Boolean = {
    val noOwnerIndex = spaceOwners.indexWhere( _ == "")
    if (noOwnerIndex == -1) {
      true
    }
    else {
      false
    }
  }

  def otherPlayerWon(agent : TicTacToeAgent, board : Vector[String]) : Boolean = {
    if (agent.name == X) {
      playerWon(O, board)
    }
    else {
      playerWon(X, board)
    }
  }

  /** Check if player X won. */
  def xWon(board : Vector[String]) : Boolean = {
    playerWon(X, board)
  }

  /** Check if player O won. */
  def oWon(board : Vector[String]) : Boolean = playerWon(O, board)

  def playerWon(agent : TicTacToeAgent, board : Vector[String]) : Boolean = playerWon(agent.name, board)

  /** Check if the given player won. */
  def playerWon(playerMark : String, board : Vector[String]) : Boolean = {
    val ownedSpaces = board.zipWithIndex.collect {
      case (element, index) if element == playerMark => index + 1
    }
    if (isWinningBoard(ownedSpaces.toVector)) {
      true
    }
    false
  }

  /** Check if the current board state is the end of a game because someone won or it's a tie. */
  def isEndState(board : Vector[String]) : Boolean = {
    if (xWon(board)) {
      true
    }
    if (oWon(board)) {
      true
    }
    if (isFullBoard(board)) {
      true
    }
    false
  }
}


class TicTacToeBoard() {
  private var spaceOwners = emptyMutableList()
  private var previousMarkMove = "" // The mark, X or O, of the last thing that was added to the board
  val uniqueBoardStates = scala.collection.mutable.Map[Vector[String], Int]()

  def emptyMutableList() : mutable.MutableList[String] = {
    mutable.MutableList.fill(Constants.boardSize){""}
  }

  def getVector() : Vector[String] = {
    spaceOwners.toVector
  }

  def setSpaceOwner(space : Int, newOwner : String) : Unit = {
    if (space < 1 || space > Constants.boardSize) {
      assert(false, s"A player tried to move to space ${space}, which is not a valid space on the Tic-tac-toe board.")
    }
    if (previousMarkMove == newOwner) {
      assert(false, s"${newOwner} tried to make a move on the board, but it was the last player to make a move.  Can't make two moves in a row.")
    }
    previousMarkMove = newOwner
    val existingOwner = spaceOwners(space - 1)
    if (existingOwner != "") {
      assert(false, s"${newOwner} tried to place someone on space ${space}, but ${existingOwner} is already there.  Board = ${spaceOwners.mkString(", ")}")
    }
    else {
      spaceOwners(space - 1) = newOwner
      if (!uniqueBoardStates.contains(spaceOwners.toVector)) {
        uniqueBoardStates(spaceOwners.toVector) = 1
      }
      else {
        uniqueBoardStates(spaceOwners.toVector) += 1
      }
    }
  }

  def resetBoard() : Unit = {
    previousMarkMove = ""
    spaceOwners = emptyMutableList()
  }
}


/** The environment is responsible for transitioning state and giving reward. */
class Environment(agent1 : TicTacToeAgent, agent2 : TicTacToeAgent) {
  val size = 3
  var spaceOwners = new TicTacToeBoard()  // Array of each space on the board with the corresponding agent name that is currently occupying the space.  0 if no one is occupying the space.

  /** Statistics to track the progress of the outcomes of episodes. */
  var xWins = 0.0
  var oWins = 0.0
  var stalemates = 0.0
  var totalGames = 0.0

  /** Clear the statistics */
  def resetGameStats() : Unit = {
    xWins = 0.0
    oWins = 0.0
    stalemates = 0.0
    totalGames = 0.0
  }

  def getOtherAgent(agent : TicTacToeAgent) : TicTacToeAgent = {
    if (agent == agent1) {
      agent2
    }
    else {
      agent1
    }
  }


  /** Make the action most recently chosen by the agent take effect. */
  def applyAction(agent : TicTacToeAgent, epsilon : Double, shouldLearn : Boolean) : Unit = {
    giveReward(agent, epsilon, shouldLearn) // For this agent's previous move that wasn't rewarded yet because the subsequent player's move could have put it into an end state
    spaceOwners.setSpaceOwner(agent.ap1, agent.name) // Take the space chosen by the agent
    debugPrint(s"${agent.name} moved to space ${agent.ap1}")
    val otherAgent = getOtherAgent(agent)
    if (isEndState(spaceOwners.getVector())) {
      giveReward(agent, epsilon, shouldLearn)
      giveReward(otherAgent, epsilon, shouldLearn)
    }
    agent.movedOnce = true
  }

  /** Determine who won and add it to the statistics */
  def countEndState() : Unit = {
    totalGames += 1
    if (xWon(spaceOwners.getVector())) {
      xWins += 1
      debugPrint("X WON!\n")
    }
    else if (oWon(spaceOwners.getVector())) {
      oWins += 1
      debugPrint("O WON!\n")
    }
    else if (isFullBoard(spaceOwners.getVector())) {
      stalemates += 1
    }
    else {
      assert(false, "It makes no sense to reach the end state and agent1 didn't win, agent 2 didn't win, and it wasn't a stalemate.")
    }
  }

  /** Update the agent's state and give it a reward for its ation. Return 1 if this is the end of the episode, 0 otherwise. */
  def giveReward(agent : TicTacToeAgent, epsilon : Double, shouldLearn : Boolean) : Unit = {
    agent.sp1 = spaceOwners.getVector()
    if (!isEndState(spaceOwners.getVector())) {
      agent.chooseAction(epsilon)
    }
    else { // The agent can not make any moves from an end state, so set the next move to 0, an invalid choice
      agent.ap1 = 0
    }
    if (shouldLearn) {
      if (playerWon(agent, spaceOwners.getVector())) {
        agent.reward(1.0)
      }
      else if (otherPlayerWon(agent, spaceOwners.getVector())) {
        agent.reward(-1.0)
      }
      else if (isFullBoard(spaceOwners.getVector())) { // Stalemate
        agent.reward(0.0)
      }
      else {
        agent.reward(0.0)
      }
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
    for (owner <- environment.spaceOwners.getVector()) {
      if (owner != "") {
        val py = rowNumber(i)
        val px = columnNumber(i)
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

}
