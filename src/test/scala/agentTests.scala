import org.scalatest.{FlatSpec, Matchers}
import tags.{CoverageAcceptanceTest, NonCoverageAcceptanceTest, UnitTest}
import ticTacToeAgents.{TicTacToeAgentNeural}
import ticTacToeEnvironment.Constants.{X, O}

class AgentSpec extends FlatSpec with Matchers {
  "The Tic-tac-toe agents" should "correctly convert a state and action into a featureVector" taggedAs(UnitTest) in {
    val neuralAgent = new TicTacToeAgentNeural(X)
    val featureVetor1 = neuralAgent.neuralNetFeatureVectorForStateAction(Vector(X, "", "", "", "", "", "" , "", ""))
    featureVetor1 should equal (Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    val featureVetor2 = neuralAgent.neuralNetFeatureVectorForStateAction(Vector(X, "", "", O, "", O, "" , "", ""))
    featureVetor2 should equal (Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0))
  }

  // TODO: Write a unit test for maxNeuralNetValueAndActionForState()
  //val tabularVRandomMinimum = 0.88
  //it should s"have greater than ${tabularVRandomMinimum*100}% X wins for Tabular vs. Random" taggedAs(CoverageAcceptanceTest) in {
    //val gameParameters = new GameParameters()
    //gameParameters.agent1Random = false
    //val ticTacToeLearning = new TicTacToeLearning(false, gameParameters)
    //val results = ticTacToeLearning.learn()
    //val xWinRatio = results._1 / results._4
    //xWinRatio should be > (tabularVRandomMinimum)
    //info(s"X won ${xWinRatio * 100.0}% of games")
    //results._4 should equal (20000.0)
    //results._5 should be > (7500)
  //}
}
