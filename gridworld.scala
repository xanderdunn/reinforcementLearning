// Gridworld problem: Starting the bottom left corner of an n by n square grid, attempt to find the optimal path to the exit in the top right corner using Q-Learning.

// The positions in the gridworld are identified by an integer p that starts 1 in the top left corner incrementing by one to the right and then continuing from the left side of the row below until it reaches the terminal state in the bottom right corner.

// TODO: Draw a rectangle at each state that maps the max Q value at that state to an increasingly dark color

// TODO later:
// - Introduce learing rate alpha
// - Introduce stochasticity into the agent's ability to act successfully
// - Intorduce a discount rate gamma
// - Introduce lambda and the ability to transition between Monte Carlo and Q-Learning
// - Introduce road blocks: positions that have an unusually negative reward
// - Introduce warps: Positions that make the agent leap to a distant and either deterministic or random location
// - Introduce multiple stop positions
// - Introduce starting position somewhere other than 1.  What about a random start position?
// - Decrease the epsilon exploration probability with time
// - Print a statistic for each episode: Number of moves taken, Optimal number of moves, # of moves that were explorations
// - Decouple the simulation from the visualization.  Redraw on every timestep rather than every action


import java.awt._
import java.awt.RenderingHints
import javax.swing._
import scala.math
import scala.util.Random


object GridWorldLearning {

  def main(args: Array[String]) {
    val frame = new JFrame("Grid World")
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    // TODO: Make the frame size a function of the grid size
    frame.setSize(350, 350)

    val gridWorld = new GridWorld()
    frame.setContentPane(gridWorld.gridWorldPanel)
    frame.setVisible(true)

    while (true) {
      val action = gridWorld.agent.act()
      gridWorld.environment.applyAction(gridWorld.agent, action)
      frame.repaint()
      Thread.sleep(10)
    }
  }

}


/** A GridWorld contains an Agent and an Environment as well as the GridWorldPanel responsible for drawing the two on screen. */
class GridWorld {
  val agent = new Agent()
  val environment = new Environment()
  val gridWorldPanel = new GridWorldPanel(this)
}

object MoveDirection extends Enumeration {
  type MoveDirection = Value
  val UP, DOWN, LEFT, RIGHT = Value
}

import MoveDirection._


/** The agent object who makes decisions on where to move. */
class Agent() {
  private var _state = 1
  private var _previousState = 1
  def state = _state
  def state_=(newState : Int): Unit = {
    _previousState = _state
    _state = newState
  }
  private var _previousAction : MoveDirection = MoveDirection.UP
  val greenStates = scala.collection.mutable.Set[Int]()
  val stateValues = scala.collection.mutable.Map[Int, scala.collection.mutable.ArrayBuffer[Int]]()  // The state-value function is stored in a map with keys that are state positions on the gridworld and values that are arrays of length 4 that store, in order, the value for each action in this state: UP, DOWN, LEFT, RIGHT

  /** Convenience method for initializing values for a given state if not already initialized */
  def getStateValues(givenState : Int) : scala.collection.mutable.ArrayBuffer[Int] = { 
    if (!stateValues.contains(givenState)) { // Initialize the state values to 0
      stateValues(givenState) = scala.collection.mutable.ArrayBuffer(0, 0, 0, 0)
    }
    return stateValues(givenState)
  }

  /** Decide what to do given the current state and return that action. */
  def act() : MoveDirection = {
    val rand = scala.util.Random
    val randomHundred = rand.nextInt(100)
    if (randomHundred <= 89) { // Exploit: Choose the greedy action and break ties randomly
      val maxValue = getStateValues(state).reduceLeft(_ max _)
      val maxValueIndices = getStateValues(state).zipWithIndex.collect {
        case (element, index) if element == maxValue => index
      }
      _previousAction = MoveDirection(maxValueIndices(rand.nextInt(maxValueIndices.size)))
    }
    else { // Explore: Randomly choose an action
      _previousAction = MoveDirection(rand.nextInt(4))
    }
    return _previousAction
  }

  /** The environment hands the agent some reward */
  def reward(value : Int) {
    // Make sure they're initialized
    getStateValues(_previousState)
    getStateValues(state)
    val updateValue = (value + stateValues(state).reduceLeft(_ max _)) - stateValues(_previousState)(_previousAction.id) // Q-Learning
    if (updateValue > 0) {
      greenStates += _previousState
      // TODO: Put a green tile at this state position if this value is non-negative
    }
    stateValues(_previousState)(_previousAction.id) += updateValue
  }

}


/** The environment is responsible for transitioning state and giving reward. */
class Environment() {
  val gridWidth = 30    // Width of each box in the grid
  val size = 8

  /** Update the agent's state give it a reward for its last ation. */
  def applyAction(agent : Agent, action : MoveDirection) {
    var tentativePosition = agent.state
    action match {
      case MoveDirection.UP => {
        if (!(agent.state <= size)) { // Can't move up in the top row
          tentativePosition -= size
        }
      }
      case MoveDirection.DOWN => {
        if (!(agent.state > size*size-size)) { // Can't move down in the bottom row
          tentativePosition += size
        }
      }
      case MoveDirection.LEFT => {
        if (agent.state % size != 1) { // Can't move left in left-most column
          tentativePosition -= 1
        }
      }
      case MoveDirection.RIGHT => {
        if (agent.state % size != 0) { // Can't move right in right-most column
        tentativePosition += 1
        }
      }
    }
    if (tentativePosition >= 1 && tentativePosition <= size*size) { // Don't allow going off the gridworld
      if (tentativePosition == size*size) {
      }
      agent.state = tentativePosition
    }
    if (tentativePosition == size*size) { // Give rewards before the state is updated
      agent.reward(0)
    }
    else {
      agent.reward(-1)
    }
    if (tentativePosition == size*size) { // Stop state reached, go back to the start state
      agent.state = 1
    }

  }

}


/** The 2D panel that's visible.  This class is responsible for all drawing. */
class GridWorldPanel(gridWorld : GridWorld) extends JPanel {
  val worldOffset = 40  // So it's not in the very top left corner

  def drawGridWorld(graphics : Graphics) {
    drawEnvironment(gridWorld.environment, graphics)
    drawGreenTiles(gridWorld.agent, gridWorld.environment, graphics)
    drawAgent(gridWorld.agent, gridWorld.environment, graphics)
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

  /** Logic to draw tiles that show the gradient of learning on the grid. */
  def drawGreenTiles(agent : Agent, environment : Environment, graphics : Graphics) {
    graphics.setColor(Color.green)
    for (state <- agent.greenStates) {
      val n = environment.size
      val x = worldOffset + columnNumber(state, n)*environment.gridWidth
      val y = worldOffset + rowNumber(state, n)*environment.gridWidth
      graphics.fillRect(x + 1, y + 1, environment.gridWidth - 2, environment.gridWidth - 2)
    }
  }

  /** Take a position in the grid and return the left to right number that is the column this position is in. */
  def columnNumber(position : Int, gridSize : Int) : Int = {
    var column = position % gridSize
    if (column == 0) {
      column = gridSize
    }
    return column - 1
  }

  /** Take a position in the grid and return the top to bottom number that is the row this position is in. */
  def rowNumber(position : Int, gridSize : Int) : Int = {
    return math.ceil(position.toDouble / gridSize.toDouble).toInt - 1
  }

  /** Logic for drawing the agent's state on the screen. */
  def drawAgent(agent : Agent, environment : Environment, graphics : Graphics) {
    val n = environment.size
    val p = agent.state
    graphics.setColor(Color.red)
    val circleDiameter = 20
    val rectangleOffset = (worldOffset/2 - circleDiameter/2)/2 // Put it in the center of the grid's box
    val py = rowNumber(p, n)
    val px = columnNumber(p, n)
    val x = worldOffset + px*environment.gridWidth + rectangleOffset
    val y = worldOffset + py*environment.gridWidth + rectangleOffset
    graphics.fillOval(x, y, circleDiameter, circleDiameter)
  }

  /** Called every time the frame is repainted. */
  override def paintComponent(graphics : Graphics) {
    drawGridWorld(graphics)
  }

}

GridWorldLearning.main(args)
