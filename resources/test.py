from random import uniform
import neurolab as nl
import numpy as np

net = nl.net.newff([[-5.0, 5.0]], [120, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

for i in range(200000):
    input = uniform(-5.0, 5.0)
    x = np.full((1, 1), input)
    y = np.sin(x) * 5.0
    # y = x
    net.train(x, y, epochs=1, show=0, goal=0.02)

maxDifference = 0.0
numOutOfRange = 0
for i in range(10000):
    input = uniform(-4.0, 4.0)
    x = np.full((1, 1), input)
    y = np.sin(x) * 5.0
    # y = x
    prediction = net.sim(x)
    withinRange = False
    difference = abs(prediction - y)
    if difference > maxDifference:
        maxDifference = difference
    if prediction < y + 0.05 and prediction > y - 0.05:
        withinRange = True
    else:
        numOutOfRange += 1
        print("With input {} got prediction {} with actual {}".format(x, prediction, y))
print("maxDifference = {}".format(maxDifference))
print("{}/10000 were out of range".format(numOutOfRange))
