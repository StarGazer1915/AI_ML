from perceptron import Perceptron

possibilities_2x = [[1, 1], [1, 0], [0, 1], [0, 0]]


p1 = Perceptron("AND", [4, 9], -1, 0.1)
print(f"{p1}\n")

p1.train(possibilities_2x, [1, 0, 0, 0], 40)
print(f"{p1}")
