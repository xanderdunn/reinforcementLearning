import org.scalatest.{FlatSpec, Matchers, ParallelTestExecution}
import ticTacToe.{TicTacToeLearning, GameParameters}

class TicTacToeSpec extends FlatSpec with Matchers with ParallelTestExecution {
  "Tic-tac-toe learning" should "have 43% X wins, 43% O wins, and 14% stalemates for Random vs. Random, regardless of being a tabular or neural net agent." in {
    val gameParametersTabularTabular = new GameParameters()
    val gameParametersTabularNeural = new GameParameters()
    gameParametersTabularNeural.agent2Tabular = false
    val gameParametersNeuralNeural = new GameParameters()
    gameParametersNeuralNeural.agent1Tabular = false
    gameParametersNeuralNeural.agent2Tabular = false
    val gameParameters = List(gameParametersTabularTabular, gameParametersTabularNeural, gameParametersNeuralNeural)
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

  var minimum = 0.90
  it should s"have greater than ${minimum*100}% X wins for Tabular vs. Random" in {
    val gameParameters = new GameParameters()
    gameParameters.agent1Random = false
    val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    val results = ticTacToeLearning.learn()
    val xWinRatio = results._1 / results._4
    xWinRatio should be > (0.87)
    info(s"X won ${xWinRatio * 100.0}% of games")
    results._4 should equal (20000.0)
    results._5 should be > (7500)
  }

  minimum = 0.85
  it should s"have greater than ${minimum*100}% X wins for Neural Net vs. Random" in {
    val gameParameters = new GameParameters()
    gameParameters.agent1Random = false
    gameParameters.agent1Tabular = false
    val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    val results = ticTacToeLearning.learn()
    val xWinRatio = results._1 / results._4
    xWinRatio should be > (0.80)
    info(s"X won ${xWinRatio * 100.0}% of games")
    results._4 should equal (20000.0)
    results._5 should be > (5750)
  }

  minimum = 0.98
  it should s"have greater than ${minimum*100}% stalemates for Tabular vs. Tabular" in {
    val gameParameters = new GameParameters()
    gameParameters.agent1Random = false
    gameParameters.agent2Random = false
    val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    val results = ticTacToeLearning.learn()
    val stalematesRatio = results._3 / results._4
    stalematesRatio should be > (0.93)
    info(s"${stalematesRatio * 100.0}% of games were stalemates")
    results._4 should equal (20000.0)
    results._5 should be > (7450)
    results._5 should be > (7500)
  }

  //it should "have greater than 90% stalemates for Neural vs. Neural" in {
    //val gameParameters = new GameParameters()
    //gameParameters.agent1Random = false
    //gameParameters.agent1Tabular = false
    //gameParameters.agent2Random = false
    //gameParameters.agent2Tabular = false
    //gameParameters.numberTrainEpisodes = 100000
    //val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    //val results = ticTacToeLearning.learn()
    //val stalematesRatio = results._3 / results._4
    //stalematesRatio should be > (0.90)
    //info(s"${stalematesRatio * 100.0}% of games were stalemates")
    //results._4 should equal (20000.0)
    //results._5 should be > (7500)
  //}
}
