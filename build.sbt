  name := "ticTacToe"
  version := "1.0"
  scalaVersion := "2.11.7"
  resolvers ++= Seq(
  )
  libraryDependencies  ++= Seq(
  )
  libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"
  wartremoverErrors ++= Warts.unsafe
