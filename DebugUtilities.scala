
package debug

object DebugUtilities {
  val DEBUG = false

  def debugPrint(message : String) {
    if (DEBUG) {
      println(message)
    }
  }
}

import DebugUtilities._
