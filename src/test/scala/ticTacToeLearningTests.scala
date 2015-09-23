// Test Infrastructure
import org.scalatest.{FlatSpec, Matchers}
import tags.{CoverageAcceptanceTest, NonCoverageAcceptanceTest, UnitTest}
// Model
import ticTacToeEnvironment.{TicTacToeLearning, GameParameters, EnvironmentUtilities}
import ticTacToeAgents.{TicTacToeAgentTypes}

class TicTacToeLearningSpec extends FlatSpec with Matchers {
  "Tic-tac-toe learning" should "have 43% X wins, 43% O wins, and 14% stalemates for Random vs. Random, regardless of being a tabular or neural net agent." taggedAs(CoverageAcceptanceTest) ignore {
    val gameParametersTabularTabular = new GameParameters(TicTacToeAgentTypes.Tabular, TicTacToeAgentTypes.Tabular, 50000, 20000)
    val gameParametersTabularNeural = new GameParameters(TicTacToeAgentTypes.Tabular, TicTacToeAgentTypes.Neural, 50000, 20000)
    val gameParametersNeuralNeural = new GameParameters(TicTacToeAgentTypes.Neural, TicTacToeAgentTypes.Neural, 50000, 20000)
    val gameParameters = Vector(gameParametersTabularTabular, gameParametersTabularNeural, gameParametersNeuralNeural)
    for (parameters <- gameParameters) {
      val ticTacToeLearning = new TicTacToeLearning(false, parameters)
      val results = ticTacToeLearning.learn()
      results._1 / results._4 should equal (0.43 +- 0.02)
      results._2 / results._4 should equal (0.43 +- 0.02)
      results._3 / results._4 should equal (0.14 +- 0.02)
      results._4 should equal (20000.0)
      results._5 should be > (8000)
    }
  }

  val tabularVRandomMinimum = 0.88
  it should s"have greater than ${tabularVRandomMinimum*100}% X wins for Tabular vs. Random" taggedAs(CoverageAcceptanceTest) in {
    val gameParameters = new GameParameters(TicTacToeAgentTypes.Tabular, TicTacToeAgentTypes.Random, 50000, 20000)
    val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    val results = ticTacToeLearning.learn()
    val xWinRatio = results._1 / results._4
    xWinRatio should be > (tabularVRandomMinimum)
    info(s"X won ${xWinRatio * 100.0}% of games")
    results._4 should equal (20000.0)
    results._5 should be > (7500)
  }

  val neuralVRandomMinimum = 0.79
  it should s"have greater than ${neuralVRandomMinimum*100}% X wins for Neural Net vs. Random" taggedAs(NonCoverageAcceptanceTest) ignore {
    val gameParameters = new GameParameters(TicTacToeAgentTypes.Neural, TicTacToeAgentTypes.Random, 100000, 20000)
    val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    val results = ticTacToeLearning.learn()
    val xWinRatio = results._1 / results._4
    xWinRatio should be > (neuralVRandomMinimum)
    info(s"X won ${xWinRatio * 100.0}% of games")
    results._4 should equal (20000.0)
    results._5 should be > (5750)
  }

  val tabularVTabularMinimum =  0.98
  it should s"have greater than ${tabularVTabularMinimum*100}% stalemates for Tabular vs. Tabular" taggedAs(CoverageAcceptanceTest) ignore {
    val gameParameters = new GameParameters(TicTacToeAgentTypes.Tabular, TicTacToeAgentTypes.Tabular, 70000, 20000)
    val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    val results = ticTacToeLearning.learn()
    val stalematesRatio = results._3 / results._4
    stalematesRatio should be > (tabularVTabularMinimum)
    info(s"${stalematesRatio * 100.0}% of games were stalemates")
    results._4 should equal (20000.0)
    results._5 should be > (7450)
    results._5 should be > (7500)
  }

  it should "have greater than 90% stalemates for Neural vs. Neural" taggedAs(NonCoverageAcceptanceTest) ignore {
    val gameParameters = new GameParameters(TicTacToeAgentTypes.Neural, TicTacToeAgentTypes.Neural, 100000, 20000)
    val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    val results = ticTacToeLearning.learn()
    val stalematesRatio = results._3 / results._4
    stalematesRatio should be > (0.90)
    info(s"${stalematesRatio * 100.0}% of games were stalemates")
    results._4 should equal (20000.0)
    results._5 should be > (7500)
  }

}
