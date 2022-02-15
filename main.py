from perceptron import Perceptron

p1 = Perceptron("AND", [0.5, 0.5], -1)
p2 = Perceptron("OR", [2, 2], -1)
p3 = Perceptron("NOT", [-1], 0.5)
p4 = Perceptron("NOR", [-1, -1, -1], 0)


print(f"\n{p1}")
p1.activation([1, 1])
p1.activation([1, 0])
p1.activation([0, 1])
p1.activation([0, 0])

print(f"\n{p2}")
p2.activation([1, 1])
p2.activation([1, 0])
p2.activation([0, 1])
p2.activation([0, 0])

print(f"\n{p3}")
p3.activation([1])
p3.activation([0])

print(f"\n{p4}")
p4.activation([1, 1, 1])
p4.activation([1, 0, 1])
p4.activation([1, 0, 0])
p4.activation([0, 0, 1])
p4.activation([1, 1, 0])
p4.activation([0, 1, 1])
p4.activation([0, 0, 0])