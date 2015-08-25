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
import scala.collection.mutable.{ArrayBuffer, Seq, IndexedSeq, MutableList, Map}
// Third Party, for Plotting
import breeze.linalg._
import breeze.plot._
// Custom
import neuralNet.NeuralNet
import neuralNet.NeuralNetUtilities._
import ticTacToe.EnvironmentUtilities._
import debug.DebugUtilities.debugPrint

package ticTacToe {

// Exception types
case class InvalidParameter(message: String) extends Exception(message)
case class InvalidCall(message: String) extends Exception(message)
case class InvalidState(message: String) extends Exception(message)


/** Update function types that can be used for learning. */
object UpdateFunctionTypes extends Enumeration {
  type UpdateFunction = Value
  // Q Learning: q(s,a) = q(s,a) + learningrate * (reward + discountRate * max_a(q(s_(t+1),a)) - q(s,a))
  // SARSA: q(s,a) = q(s,a) + learningrate * (reward + discountRate * max_a(q(s_(t+1),a)) - q(s,a))
  val SARSA, QLearning = Value
}


/** Parameters for the Q value update function and the neural network. */
object Parameters {
  // Tabular Parameters
  val tabularAlpha = 0.1
  // Both
  val epsilon = 0.2
  val gamma = 1.0 // discount rate
  // Neural Net Parameters
  val neuralNetAlpha = 0.5             // The learning rate in the neural net itself
  val neuralInitialBias = 0.33  // This is in the range [0, f(n)] where n is the number of input neurons and f(x) = 1/sqrt(n).   See here: http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
  val neuralNumberHiddenNeurons = 40
  val neuralValueLearningAlpha = 1.0/neuralNumberHiddenNeurons // The learning rate used by the value update function
  val updateFunction = UpdateFunctionTypes.SARSA
}


/** Object to store game-specific parameters.  This is passed to TicTacToeLearning to start a game. */
class GameParameters {
  var agent1Tabular = true
  var agent2Tabular = true
  var agent1Random = true
  var agent2Random = true
  var numberTrainEpisodes = 50000
  var numberTestEpisodes = 20000
}

/** Executed to initiate playing and learning Tic-tac-toe. */
class TicTacToeLearning(generateGraph : Boolean, gameParameters : GameParameters, showVisual : Boolean = false) {
  /** The go button.  Actually initiates game play and learning until the ultimate goal is achieved: Generate a graph or return results in the form (# games X won, # games O won, # stalemates, # total games, # unique board states encountered). */
  def learn() : (Double, Double, Double, Double, Int) = {
    if (generateGraph) { // Set to true if you want to generate graphs instead of initiating single test runs with output in the terminal
      PlotGenerator.generateLearningCurves() // TODO: Use the below code path rather than a separate code path for collecting data to plot on this run.
      System.exit(0)
    }

    val ticTacToeWorld = new TicTacToeWorld(gameParameters.agent1Tabular, gameParameters.agent2Tabular, gameParameters.agent1Random, gameParameters.agent2Random)
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
      playEpisode(ticTacToeWorld, Parameters.epsilon, "")
    }
    environment.resetGameStats()
    while (environment.totalGames < gameParameters.numberTestEpisodes) {
      playEpisode(ticTacToeWorld, 0.0, "")
    }
    val uniqueBoardStates = ticTacToeWorld.environment.spaceOwners.uniqueBoardStates
    (environment.xWins, environment.oWins, environment.stalemates, environment.totalGames, uniqueBoardStates.size)
  }

  object PlotGenerator {
    def generateLearningCurves() {
      val settings = List((25000, 200, true, false, true, s"Tabular vs. Random Agent, epsilon=${Parameters.epsilon}  alpha=${Parameters.tabularAlpha}", "tabularVrandom.pdf", 1),
        (50000, 100, false, false, true, s"Neural vs. Random Agent, epsilon=${Parameters.epsilon} learningAlpha=${Parameters.neuralValueLearningAlpha} netAlpha=${Parameters.neuralNetAlpha} gamma=${Parameters.gamma} ${Parameters.neuralNumberHiddenNeurons} hidden neurons ${Parameters.neuralInitialBias} initialBias", "neuralVrandom.pdf", 1),
        (4000, 150, true, false, false, s"Tabular vs. Tabular, epsilon=${Parameters.epsilon}  alpha=${Parameters.tabularAlpha}", "tabularVtabular.pdf", 2),
        (40000, 100, false, false, false, s"Neural vs. Neural, epsilon=${Parameters.epsilon} learningAlpha=${Parameters.neuralValueLearningAlpha} netAlpha=${Parameters.neuralNetAlpha} gamma=${Parameters.gamma} ${Parameters.neuralNumberHiddenNeurons} hidden neurons ${Parameters.neuralInitialBias} initial bias", "neuralVneural.pdf", 3))

      for (setting <- settings) {
        val numberEpisodes = setting._1
        val numberIterations = setting._2
        val tabular = setting._3
        val playerXRandom = setting._4
        val playerORandom = setting._5
        val title = setting._6
        val filename = setting._7
        val plotting = setting._8 // 1 if I'm plotting player X wins 2 if I'm plotting stalemates and 3 if I'm plotting both player 3 and player 4 wins

        var i = 0
        val episodeNumbers : Seq[Double] = Seq.fill(numberEpisodes){0.0}
        while (i < numberEpisodes) {
          episodeNumbers(i) = i.toDouble + 1.0
          i += 1
        }
        var iteration = 0
        val finalResults1 : Seq[Double] = Seq.fill(numberEpisodes){0.0}
        val finalResults2 : Seq[Double] = Seq.fill(numberEpisodes){0.0}
        while (iteration < numberIterations) {
          println(s"Iteration ${iteration}/${numberIterations}")
          val results = playTrainingSession(numberEpisodes, tabular, playerXRandom, playerORandom, Parameters.epsilon)
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

        val f = Figure()
        val p = f.subplot(0)
        if (plotting == 1 || plotting == 2) {
          p += plot(episodeNumbers, finalResults1, '.')
        }
        else if (plotting == 3) {
          p += plot(episodeNumbers, finalResults1, '.')
          p += plot(episodeNumbers, finalResults2, '.')
        }
        p.xlabel = "Episodes"
        if (plotting == 1 || plotting == 3) {
          p.ylabel = s"% wins out of ${numberIterations.toInt} iterations"
        }
        else {
          p.ylabel = s"% stalemates out of ${numberIterations.toInt} iterations"
        }
        p.title = title
        f.saveas(filename)
      }
    }
  }

  /**  Play a set of episodes for training. */
  // TODO: This can be removed entirely by absorbing it into the learn() method above when the generateGraph boolean is set
  def playTrainingSession(numberEpisodes : Int, tabular : Boolean, playerXRandom : Boolean, playerORandom : Boolean, epsilon : Double) : Seq[Double] = {
    println(s"Playing a training session with ${numberEpisodes} episodes")
    val stats : IndexedSeq[Double] = IndexedSeq.fill(numberEpisodes){0.0}
    var episodeCounter = 0
    val ticTacToeWorld = new TicTacToeWorld(tabular, tabular, playerXRandom, playerORandom)
    while (episodeCounter < numberEpisodes) {
      stats(episodeCounter) = playEpisode(ticTacToeWorld, epsilon, "X")
      episodeCounter += 1
    }
    return stats
  }

  def playEpisode(ticTacToeWorld : TicTacToeWorld, epsilon : Double, collectingDataFor : String) : Double = {
    var episodeOutcome = -2.0
    while (episodeOutcome == -2.0) { // Train with epsilon
      episodeOutcome = iterateGameStep(ticTacToeWorld, epsilon, None, collectingDataFor)
    }
    episodeOutcome = -2.0
    while (episodeOutcome == -2.0) { // Test run with epsilon = 0
      episodeOutcome = iterateGameStep(ticTacToeWorld, 0.0, None, collectingDataFor)
    }
    return episodeOutcome
  }

  /** Take one step in the game: The player takes an action, the other player responds, and the board hands out reward to both players.  If you're collecting data for a graph, pass in the string "X" or "O" for the player whose data you're interested in.  This method returns 1 if that player won this episode, -1 if it lost, 0 if it was a stalemate, and -2 if the episode hasn't ended. */
  def iterateGameStep(ticTacToeWorld : TicTacToeWorld, epsilon : Double, frame : Option[JFrame], collectingDataFor : String) : Double = {
    val agent = ticTacToeWorld.currentPlayer
    val environment = ticTacToeWorld.environment
    environment.applyAction(agent, epsilon)
    var returnValue = -2.0
    if (environment.isEndState()) {
      if (environment.playerWon(ticTacToeWorld.agent1)) {
        returnValue = 1.0
      }
      else if (environment.playerWon(environment.getOtherAgent(ticTacToeWorld.agent1))) {
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
    return returnValue
    // TODO: Show some text on the tic tac toe board when a certain player wins
    // TODO: Fix the timing such that the end state is visible to the user for a moment.
    //Thread.sleep(500)
  }
}


/** A TicTacToeWorld contains an Agent and an Environment as well as the TicTacToePanel responsible for drawing the two on screen. */
class TicTacToeWorld(_agent1Tabular : Boolean, _agent2Tabular : Boolean, agent1Random : Boolean, agent2Random : Boolean) {
  def agent1Tabular = _agent1Tabular
  def agent2Tabular = _agent2Tabular
  val agent1 = new Agent("X", agent1Tabular, agent1Random)
  val agent2 = new Agent("O", agent2Tabular, agent2Random)
  val agents = List(agent1, agent2)
  val environment = new Environment(agent1, agent2)
  val ticTacToePanel = new TicTacToePanel(this)
  var currentPlayer = agent1
  var firstPlayer = agent1
  val xLostStates = scala.collection.mutable.Map[List[String], Int]()

  /** Reset the agent and states for a new episode */
  def endEpisode() {
    currentPlayer = agents(nextInt(agents.size))
    debugPrint(s"firstPlayer = ${firstPlayer.name}")
    environment.spaceOwners.resetBoard()
    agent1.s = List.fill(9){""}
    agent1.sp1 = List.fill(9){""}
    agent1.movedOnce = false
    agent1.a = 0
    agent1.ap1 = 0
    agent2.s = List.fill(9){""}
    agent2.sp1 = List.fill(9){""}
    agent2.movedOnce = false
    agent2.a = 0
    agent2.ap1 = 0
  }
}


/** The agent object who makes decisions on where to places X's and O's.  Because there are two players, players are identified by an integer value. */
class Agent(_name : String, _tabular : Boolean, _random : Boolean) {
  val name = _name
  private var _sp1 : List[String] = List.fill(9){""}
  var s = List.fill(9){""}
  def sp1 : List[String] = _sp1
  def sp1_=(newState : List[String]) : Unit = {
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
  val stateValues = Map[List[String], Map[Int, Double]]()  // The state-value function is stored in a map with keys that are environment states of the Tic-tac-toe board and values that are arrays of the value of each possible action in this state.  A possible action is any space that is not currently occupied.
  def tabular = _tabular
  //val neuralNet = new NeuralNet(10, Parameters.neuralNumberHiddenNeurons, Parameters.neuralNetAlpha, Parameters.neuralInitialBias)
  val neuralNets = {
    val mutableNeuralNets = Map[Int, NeuralNet]()
    for (i <- 1 to 9) {
      mutableNeuralNets(i) = new NeuralNet(18, Parameters.neuralNumberHiddenNeurons, Parameters.neuralNetAlpha, Parameters.neuralInitialBias)
    }
    mutableNeuralNets
  }
  def random = _random
  var movedOnce = false // To know not to update the value function before its first action

  /** Convenience method for initializing values for a given state if not already initialized */
  def getStateValues(state : List[String]) : Map[Int, Double] = {
    if (!stateValues.contains(state)) { // Initialize the state values to 0
      if (isFullBoard(state)) {  // The state values in the stop state are always 0, so always return a map full of zeros
        val zeroMap = Map[Int, Double]()
        for (i <- 1 until 10) {
          zeroMap(i) = 0.0
        }
        stateValues(state) = zeroMap
      }
      else {
        val newStateValues = Map[Int, Double]()
        for (emptySpace <- emptySpaces(state)) {
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
    if (possibleMoves.size == 0) { // The value of an end state position is always 0, and there is no position to take next
      return (0.0, 0)
    }
    debugPrint(s"${name} is getting max neural net values for spaces ${possibleMoves.mkString(", ")}")
    var maxValue = Double.MinValue
    var greedyAction = 0
    val stateValues = Map[Int, Double]()
    for (possibleMove <- possibleMoves) {
      val input = neuralNetFeatureVectorForStateAction(state)
      val value = neuralNets(possibleMove).feedForward(input.toArray)
      stateValues(possibleMove) = value
      if (value > maxValue || maxValue == Double.MinValue) {
        greedyAction = possibleMove
        maxValue = value
      }
    }
    debugPrint(s"Player is choosing state values from ${stateValues}")
    val maxValueSpaces = ArrayBuffer[Int]()
    for ((key, value) <- stateValues) {
      if (value == maxValue) {
        maxValueSpaces += key
      }
    }
    if (maxValueSpaces.size > 1) {
      debugPrint(s"Have max value state ties on states ${maxValueSpaces.mkString(", ")}")
    }
    return (maxValue, maxValueSpaces(scala.util.Random.nextInt(maxValueSpaces.size))) // Break ties randomly
  }


  /** The agent chooses the next action to take. */
  def chooseAction(epsilon : Double) {
    if (epsilon < 0.0 || epsilon > 1.0) {
      throw new InvalidParameter(s"epsilon = ${epsilon} was passed in, but it only makes sense if it's greater than 0.0 and less than 1.0.")
    }
    if (emptySpaces(sp1).size == 0) {
      a = ap1 // If the agent is in a stop state, simply update the action that will be used to give reward
      return
    }
    if (_random) {
      val prospectiveSpaces = emptySpaces(sp1)
      ap1 = prospectiveSpaces(nextInt(prospectiveSpaces.size))
    }
    else {
      val randomHundred = nextInt(100)
      if (randomHundred <= (100 - (epsilon * 100.0) - 1)) { // Exploit: Choose the greedy action and break ties randomly
        if (tabular) {
          ap1 = tabularGreedyAction(sp1)
        }
        else {
          ap1 = neuralNetGreedyAction(sp1)
        }
      }
      else { // Explore: Randomly choose an action
        val prospectiveSpaces = emptySpaces(sp1)
        ap1 = prospectiveSpaces(nextInt(prospectiveSpaces.size))
      }
    }
    if (a == 0) { // Actions are always applied from a, so if a is not set to anything, then the "next action" ap1 is not actually known yet, and we set the "previous action" to the same value.
      a = ap1
    }
  }

  /** Use a neural network to choose the greedy action to take */
  def neuralNetGreedyAction(boardState : List[String]) : Int = {
    return maxNeuralNetValueAndActionForState(boardState)._2
  }

  /** Decide what to do given the current environment and return that action. */
  def tabularGreedyAction(boardState : List[String]) : Int = {
    val stateValues = getStateValues(boardState)
    val maxValue = stateValues.maxBy(_._2)._2
    val maxValueSpaces = ArrayBuffer[Int]()
    for ((key, value) <- stateValues) {
      if (value == maxValue) {
        maxValueSpaces += key
      }
    }
    return maxValueSpaces(nextInt(maxValueSpaces.size))
  }

  def sanityCheckReward(reward : Double) {
    if (a == 0 || ap1 == 0) {
      throw new InvalidCall(s"An attempt was made to give reward to ${name} while its previous action is ${a}. A player must move at least once to be rewarded for it.")
    }
    if (!emptySpaces(s).contains(a)) {
      throw new InvalidState(s"${name} is being rewarded for (s, a) (${s}, ${a}), but it isn't possible to take that action in that given state.")
    }
    if (!isFullBoard(sp1) && !emptySpaces(sp1).contains(ap1)) {
      throw new InvalidState(s"${name} is being rewarded for (sp1, ap1) (${sp1}, ${ap1}), but it isn't possible to take that action in that given state.")
    }
    val previousAndCurrentStateDifferences = differenceBetweenBoards(s, sp1)
    if (!isFullBoard(sp1)) { // Check that the state is paired with an action that's possible (It's not possible to take an action that's already occupied)
      if (previousAndCurrentStateDifferences.size != 2 || !previousAndCurrentStateDifferences.contains("X") || !previousAndCurrentStateDifferences.contains("O")) {
        if (reward == 0.0) {
          throw new InvalidState(s"${name} is being given reward ${reward} for moving from ${s} ${sp1}")
        }
      }
    }
  }

  /** On any value update, the value should approach the expected return of the state action pair, not the immediate reward. The target is the reward plus the value of the next state. */
  def sanityCheckValueUpdate(reward : Double, previousValue : Double, updatedValue : Double, targetValue : Double) {
    val target = reward + targetValue
    if (Math.abs(previousValue - target) < Math.abs(updatedValue - target)) {  // The difference between the target and the previous value should always be greater than the difference between the target and updated value
      throw new InvalidState(s"Player ${name} received a reward of ${reward} and the state value was updated from ${previousValue} to ${updatedValue}.  However, it was expected that the value would get closer to the reward ${reward} + target value ${targetValue}")
    }
  }

  def tabularReward(reward : Double) {
    // Make sure they're initialized
    getStateValues(s)
    getStateValues(sp1)
    val previousStateValue = stateValues(s)(a)
    var updateValue = 0.0
    var targetValue = 0.0
    Parameters.updateFunction match {
      case UpdateFunctionTypes.SARSA => {
        targetValue = stateValues(sp1)(ap1)
        updateValue = (Parameters.tabularAlpha)*(reward + Parameters.gamma * stateValues(sp1)(ap1) - stateValues(s)(a))
      }
      case UpdateFunctionTypes.QLearning => {
        debugPrint(s"alpha = ${Parameters.tabularAlpha} reward = ${reward} maxStateValue = ${stateValues(sp1).maxBy(_._2)._2} previousStateValue = ${stateValues(s)(a)}")
        targetValue = stateValues(sp1).maxBy(_._2)._2
        updateValue = (Parameters.tabularAlpha)*(reward + Parameters.gamma * stateValues(sp1).maxBy(_._2)._2 - stateValues(s)(a))
      }
    }
    stateValues(s)(a) += updateValue
    sanityCheckValueUpdate(reward, previousStateValue, stateValues(s)(a), targetValue)
  }

  def neuralReward(reward : Double) {
    debugPrint(s"Updating ${name}'s neural net for making the move ${a} from the state ${s}")
    val previousStateFeatureVector = neuralNetFeatureVectorForStateAction(s)
    val previousStateActionValue = neuralNets(a).feedForward(previousStateFeatureVector)
    var updateValue = 0.0
    var targetValue = 0.0
    Parameters.updateFunction match {
      case UpdateFunctionTypes.SARSA => {
        val stateFeatureVector = neuralNetFeatureVectorForStateAction(sp1)
        val stateActionValue = neuralNets(ap1).feedForward(stateFeatureVector)
        targetValue = stateActionValue
        updateValue = previousStateActionValue + Parameters.neuralValueLearningAlpha * (reward + Parameters.gamma * stateActionValue - previousStateActionValue)
        debugPrint(s"previousStateActionValue = ${previousStateActionValue} alpha = ${Parameters.neuralValueLearningAlpha} reward = ${reward} gamma = ${Parameters.gamma} stateActionValue = ${stateActionValue} updateValue = ${updateValue}")
      }
      case UpdateFunctionTypes.QLearning => {
        val stateMaxValue = maxNeuralNetValueAndActionForState(sp1)._1
        targetValue = stateMaxValue
        updateValue = previousStateActionValue + Parameters.neuralValueLearningAlpha * (reward + Parameters.gamma * stateMaxValue - previousStateActionValue)
      }
    }
    neuralNets(a).train(previousStateFeatureVector, updateValue)
    debugPrint(s"Updated player ${name}'s neural net for ${previousStateFeatureVector.mkString(", ")} with reward ${reward} and targetValue ${updateValue}")
    val previousStateActionValueUpdated = neuralNets(a).feedForward(previousStateFeatureVector)
    sanityCheckValueUpdate(reward, previousStateActionValue, previousStateActionValueUpdated, targetValue)
  }


  /** The environment calls this to reward the agent for its action. */
  def reward(reward : Double) {
    if (movedOnce && !random) { // There's no need to update if the agent is a random player or if the agent hasn't moved yet.
      sanityCheckReward(reward)
      debugPrint(s"Give reward ${reward} to ${name} moving from ${s} with action ${a} to ${sp1}")
      if (tabular) {
        tabularReward(reward)
      }
      else {
        neuralReward(reward)
      }
    }
  }

}


/** Static convenience functions for handling the environment. */
object EnvironmentUtilities {
  val size = 3
  /** Return all spaces where the given agent currently has its mark. */
  def spacesOccupiedByAgent(agent : Agent, spaceOwners : List[String]) : List[Int] = {
    return spaceOwners.zipWithIndex.collect {
      case (element, index) if element == agent.name => index + 1
    }
  }

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
    if ((spaces.contains(1) && spaces.contains(5) && spaces.contains(9)) || (spaces.contains(7) && spaces.contains(5) && spaces.contains(3))) {
      return true
    }
    else {
      return false
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

  /** Compares two boards and returns the values in boardState2 that are different from boardState1 */
  def differenceBetweenBoards(boardState1 : List[String], boardState2 : List[String]) : List[String] = {
    var i = 0
    val differences = MutableList[String]()
    for (board1Position <- boardState1) {
      val board2Position = boardState2(i)
      if (board1Position != board2Position) {
        differences += board2Position
      }
      i += 1
    }
    return differences.toList
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
  private var previousMarkMove = "" // The mark, X or O, of the last thing that was added to the board
  val uniqueBoardStates = scala.collection.mutable.Map[List[String], Int]()

  def emptyMutableList() : MutableList[String] = {
    return MutableList.fill(9){""}
  }

  def getList() : List[String] = {
    return spaceOwners.toList
  }

  def setSpaceOwner(space : Int, newOwner : String) {
    if (space < 1 || space > 9) {
      throw new InvalidParameter(s"A player tried to move to space ${space}, which is not a valid space on the Tic-tac-toe board.")
    }
    if (previousMarkMove == newOwner) {
      throw new InvalidState(s"${newOwner} tried to make a move on the board, but it was the last player to make a move.  Can't make two moves in a row.")
    }
    previousMarkMove = newOwner
    val existingOwner = spaceOwners(space - 1)
    if (existingOwner != "") {
      throw new InvalidCall(s"${newOwner} tried to place someone on space ${space}, but ${existingOwner} is already there.  Board = ${spaceOwners.mkString(", ")}")
    }
    else {
      spaceOwners(space - 1) = newOwner
      if (!uniqueBoardStates.contains(spaceOwners.toList)) {
        uniqueBoardStates(spaceOwners.toList) = 1
      }
      else {
        uniqueBoardStates(spaceOwners.toList) += 1
      }
    }
  }

  def resetBoard() {
    previousMarkMove = ""
    spaceOwners = emptyMutableList()
  }
}


/** The environment is responsible for transitioning state and giving reward. */
class Environment(agent1 : Agent, agent2 : Agent) {
  val size = 3
  var spaceOwners = new TicTacToeBoard()  // Array of each space on the board with the corresponding agent name that is currently occupying the space.  0 if no one is occupying the space.

  def otherPlayerWon(agent : Agent) : Boolean = {
    if (agent.name == "X") {
      return playerWon("O")
    }
    else {
      return playerWon("X")
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

  def playerWon(agent : Agent) : Boolean = {
    return playerWon(agent.name)
  }

  /** Check if the given player won. */
  def playerWon(playerMark : String) : Boolean = {
    val ownedSpaces = spaceOwners.getList().zipWithIndex.collect {
      case (element, index) if element == playerMark => index + 1
    }
    if (isWinningBoard(ownedSpaces.toList)) {
      return true
    }
    return false
  }

  /** Check if the current board state is the end of a game because someone won or it's a tie. */
  def isEndState() : Boolean = {
    if (xWon()) {
      return true
    }
    if (oWon()) {
      return true
    }
    if (isFullBoard(spaceOwners.getList())) {
      return true
    }
    return false
  }

  /** Statistics to track the progress of the outcomes of episodes. */
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

  def getOtherAgent(agent : Agent) : Agent = {
    if (agent == agent1) {
      return agent2
    }
    else {
      return agent1
    }
  }

  /** Make the action most recently chosen by the agent take effect. */
  def applyAction(agent : Agent, epsilon : Double) {
    giveReward(agent, epsilon) // For this agent's previous move that wasn't rewarded yet because the subsequent player's move could have put it into an end state
    spaceOwners.setSpaceOwner(agent.ap1, agent.name) // Take the space chosen by the agent
    debugPrint(s"${agent.name} moved to space ${agent.ap1}")
    val otherAgent = getOtherAgent(agent)
    if (isEndState()) {
      giveReward(agent, epsilon)
      giveReward(otherAgent, epsilon)
    }
    agent.movedOnce = true
  }

  /** Determine who won and add it to the statistics */
  def countEndState() {
    totalGames += 1
    if (xWon()) {
      xWins += 1
      debugPrint("X WON!\n")
    }
    else if (oWon()) {
      oWins += 1
      debugPrint("O WON!\n")
    }
    else if (isFullBoard(spaceOwners.getList())) {
      stalemates += 1
    }
    else {
      throw new InvalidState("It makes no sense to reach the end state and agent1 didn't win, agent 2 didn't win, and it wasn't a stalemate.")
    }
  }

  /** Update the agent's state and give it a reward for its ation. Return 1 if this is the end of the episode, 0 otherwise. */
  def giveReward(agent : Agent, epsilon : Double) {
    agent.sp1 = spaceOwners.getList()
    agent.chooseAction(epsilon)
    if (playerWon(agent)) {
      agent.reward(1.0)
    }
    else if (otherPlayerWon(agent)) {
      agent.reward(-1.0)
    }
    else if (isFullBoard(spaceOwners.getList())) { // Stalemate
      agent.reward(0.0)
    }
    else {
      agent.reward(0.0)
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
