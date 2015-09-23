// Standard Library
import scala.math
import scala.util.Random.{nextInt, nextDouble}
// Custom
import debug.DebugUtilities.debugPrint

package learningFunctions {

/** Update function types that can be used for learning. */
object UpdateFunctionTypes extends Enumeration {
  type UpdateFunction = Value
  // Q Learning: q(s,a) = q(s,a) + learningrate * (reward + discountRate * max_a(q(s_(t+1),a)) - q(s,a))
  // SARSA: q(s,a) = q(s,a) + learningrate * (reward + discountRate * max_a(q(s_(t+1),a)) - q(s,a))
  val SARSA, QLearning = Value
}

}
