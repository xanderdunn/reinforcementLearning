# Osaro Deep Reinforcement Learning
Implement a deep reinforcement learner on the Tic-tac-toe problem.  No outside help was enlisted other than the typical Google searches for Scala syntax, equations, etc.  I made heavy reference of Sutton and Barto's _Reinforcement Learning_.

## Run
### Tic-tac-toe
Because of package dependencies on breeze and ScalaTest, building now uses `sbt`.  On Mac you can install it using HomeBrew with `brew install sbt`.  Then, in the project's directory:
- `sbt run`
- Expect to see both output in the terminal as well as a window that will pop up to visualize the problem.  If you want to see the visualization occurring at human pace, open ticTacToe.scala and uncomment the line `Thread.sleep(500)`

## What's New
- Tabular vs. tabular, neural net vs. neural net, and neural net vs. tabular game situations have been added.  
    - The tabular vs. tabular performs optimally and reaches nearly ~99% stalemates in 50000 episodes
    - The neural net vs. neural net underperforms and converges on 50% X wins and 50% O wins.
- As a result of bug fixes, good results are achieved on the tabular vs. random, tabular vs. tabular, and neural vs. random situations with parameters that make more sense than in the first submission.  For the neural net, the learning rate is now 1/(number neurons).  The reward function is now 1 for a win and -1 for a loss.  All other moves receive a reward of 0.  This is as it should be and indicates learning occurring as expected.
- Due to considerably less than optimal results on neural vs. neural, as sanity checks I: 
    - Represented the feature vector as 19 inputs (9 spaces for X, 9 spaces for O, 1 action) rather than 10 (9 spaces on the board + 1 action).  As expected, this had no effect on results.
    - Used 9 neural networks per agent rather than 1.  With this configuration, the feature vector is simply the 9 input positions and each neural net is specific to a given action.  This actually had a positive effect on results.  This makes sense in that it decreases the complexity of the function any given neural network needs to approximate, but it should be unnecessary.
    - With some extreme parameters, I'll see the neural vs. neural situation get up to 100% stalemates: epsilon=0.1, learningAlpha=1/(number hidden neurons), neuralNetAlpha=0.7, gamma=0.9, numberHiddenNeurons=26, bias=0.0, initialWeights=0.25 and using the bipolar sigmoid activation function instead of sigmoid.  However, the results are very unstable and on a subsequent run might simply converge to 50% wins/50% losses again.  I could potentially output the weights from one of these runs that manages to reach 100% stalemates and then simply instantiate future neural nets from those weights.  However, the lack of reproducible learning from the net is unsatisfactory.
    - Introduced numerous exceptions on invalid situations, catching a number of bugs
    - Added unit tests on the neural network to show that it accurately approximates f(x) = x and f(x) = sin(x).  The unit tests can be run with `sbt test`

## Graphs
- See the convergence graphs in the folder ./graphs.  The next step for these is to convert them to an average with error bars rather than a scatter plot.  
- When collecting data for the graphs, instead of collecting from the training data where epsilon is non-zero, I should train for some number of episodes, turn off epsilon and test/collection data, and then repeat to the end.
- If you want to generate the graphs for yourself, change the check at the top of ticTacToeWorld.scala `if (false) { PlotGenerator.generateLearningCurves() }`.  This will generate graphs in the current directory instead of testing a single run with output in the terminal.

## Challenges
- In my first submission, where the players only played against a random player, player X was always the start player.  I discovered that this gives player X a slight advantage.  This can be seen by playing a random agent vs. random agent where X is always the start player.  X will win 58% of the time.  Hence, at the beginning of each episode the starting player is now randomly chosen.  With a randomly chosen start player, X and O will win 43% of the time and 14% will be stalemates.
- Switching over to player vs. player revealed a number of bugs in my implementation that were insignificant in the player vs. random case but became significant against an intelligent opponent.  For example, my code would sometimes pair the wrong previous state with a certain action when updating the value function.  In the simple agent vs. random case where the agent always starts, this noise was apparently not a problem.  It became a problem in agent vs. agent.
- In the unit tests ticTacToeTests.scala you can see tests that show the neural network performs well learning the functions f(x) = x and f(x) = sin(x).  This would indicate that my neural net is functioning properly, but perhaps not best suited to this situation.  At least not with the configurations I've tried.

## Future Directions
- I believe the neural net is simply getting stuck in local optima, as it does achieve 100% stalemate rate *sometimes* under certain parameters.  My next step is to try annealing to avoid this.
- I'd like to train the tabular vs. tabular situation, output the tabular value function table, and then use that as training data for my neural network.  Then I can test to see how accurately the neural net is capable of representing the value function in question.  This will separate the problems of the neural net's ability to approximate the function with the neural net + reinforcement learner's ability to traverse states sufficiently to train itself. 
- It would probably be a good idea to hook up my reinforcement learner to a more established neural network library so that I can rapidly test various configurations (more than 1 hidden layer, various activation functions, etc.).  This will sanity check my neural net implementation as well as allow me to quickly discover what might work better.

