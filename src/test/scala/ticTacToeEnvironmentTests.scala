// Test Infrastructure
import org.scalatest.{FlatSpec, Matchers}
import tags.{CoverageAcceptanceTest, NonCoverageAcceptanceTest, UnitTest}
// Model
import ticTacToeEnvironment.{TicTacToeLearning, GameParameters, EnvironmentUtilities}
import ticTacToeEnvironment.Constants.{X, O}
import ticTacToeAgents.{TicTacToeAgentTypes}

class TicTacToeEnvironmentSpec extends FlatSpec with Matchers {
  "The Tic-tac-toe environment" should "correctly identify a winning board" taggedAs(UnitTest) in {
    EnvironmentUtilities.isWinningBoard(Vector(1, 1, 1, 0, 0, 0, 0, 0, 0)) should be (true)
    EnvironmentUtilities.isWinningBoard(Vector(0, 0, 0, 0, 0, 0, 0, 0, 0)) should be (false)
  }

}
