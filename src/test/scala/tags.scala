import org.scalatest.Tag

package tags {

object UnitTest extends Tag("xander.tags.UnitTest")                                    // A unit test tests a single unit, often a single function, for its most basic functionality.  Unit tests should be light and execute very quickly.
object CoverageAcceptanceTest extends Tag("xander.tags.CoverageAcceptanceTest")        // An acceptance test is a more comprehensive behavioral test that ensures functionality and performance requirements are met.
object NonCoverageAcceptanceTest extends Tag("xander.tags.NonCoverageAcceptanceTest")  // These are the same as above, but are deemed too computationally intensive to run under the instrumented coverage code, so they're excluded from coverage calculations and simply run as normal tests.

}
