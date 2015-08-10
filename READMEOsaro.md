# Osaro Deep Reinforcement Learning
Implement a deep reinforcement learner on the Tic-tac-toe problem.  No outside help was enlisted other than the typical Google searches for Scala syntax, equations, etc.  I made heavy reference of Sutton and Barto's _Reinforcement Learning_.

## Source Code
- ticTacToe/ticTacToe.scala is the problem that you requested.  When you run it, you should see:
    - Training and test results on a tabular Q Learner.  Against a random player, it wins 98% of 100000 games after trained on 100000 games.
    - Training and test results on a Q Learner whose value function is a neural network using a sigmoid activation function with 10 inputs, 1 hidden layer with 26 neurons, and 1 output.  Against a random player, it wins ~95% of 100000 games after trained on 200000 games.
- neuralNetParameterTuning.csv = Results from testing various parameter combinations of the neural net.  If I were to spend more time on this, I'd automate the selection of optimal parameters.
- gridworld/girdworld.scala = I've thrown this in as a bonus.  I implemented it recently.  It's a tabular Q Learning implementation of the n*n gridworld problem.  

## Run
### Tic-tac-toe
- `cd ticTacToe`
- `scalac -d build *.scala`
- `scala -classpath build TicTacToeLearning`
- Expect to see both output in the terminal as well as a window that will pop up to visualize the problem.  If you want to see the visualization occurring at human pace, open ticTacToe.scala and uncomment the line `Thread.sleep(500)`
### Gridworld
- `cd gridworld`
- `scalac -d build *.scala`
- `scala -classpath build GridWorldLearning`
- Although there is no detailed terminal output, you should visually see it go from a randomly crawler to converge on the optimal solution over time.

## Notes
- I wrote 100% of this code myself, without any third party libraries other than the Scala standard library.  I chose to write my own Q Learning and neural network implementations for the sake of the exercise.  If my goal were production ready code, instead of reinventing the wheel less I'd use existing frameworks (Theano, Cafe, ...) that are battle hardened + CUDA + parallelized, etc.
- Why Scala? Most of the machine learning companies I've spoken to have expressed interest in Scala, mainly because of Spark.  I've never before written anything in Scala, so I figured I'd make it a learning experience.  
- Quality: I don't consider this code production ready.  My goal was speed, not quality.  I'd want to redesign much of the architecture and implement a lot more unit tests if this were going into production anywhere.

## Challenges
- The largest portion of time was spent on debugging the game model and the neural network.  The RL itself was fairly straightforward.
- The neural network function approximator seems to sometimes get stuck in sub-optimal solutions.  When I run ticTacToe.scala, about 1/5 runs will have a win rate of ~87%.  The others will consistently be ~95%.  I was able to decrease this by decreasing the neural net's learning rate.  However, it still occurs.  I'd be interested in figuring out why.

## Future Directions
- The neural net reinforcement learner has a win rate ~4% lower than the tabular learner.  I'd like to experiment with many different combinations of activation functions, layer counts, neuron counts, initial weights and biases, learning rates, discount rates, etc. in an attempt to close this gap.  The neural network learner is also a much slower learner.
- Throughout the code you'll also see TODOs with future intentions, including: Add SARSA, implement agent vs. agent play, etc.  I'll probably flesh these out over the coming days.  I'm excited to keep working on it and apply it to a more interesting problem!

