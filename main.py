from perceptron import Perceptron, PerceptronLayer, PerceptronNetwork

# =============== DEFINING PERCEPTRONS =============== #
p1 = Perceptron("AND", [0.5, 0.5], -1)
p2 = Perceptron("OR", [0.1, 0.1], -0.1)
p3 = Perceptron("NOT", [-1], 0.5)
p4 = Perceptron("Figuur 2.8", [0.6, 0.3, 0.2], -0.4)
p5 = Perceptron("NOR 3x", [-1, -1, -1], 0)
p6 = Perceptron("NOR 2x", [-1, -1], 0)
p7 = Perceptron("NAND", [-0.1, -0.1], 0.1)
p8 = Perceptron("PASSTROUGH", [0, 0, 1], 0)
p9 = Perceptron("HA-AND", [0.5, 0.5, 0], -1)

# =============== DEFINING NETWORK [XOR] =============== #
pLayer1_1 = PerceptronLayer([p7, p2])
pLayer1_2 = PerceptronLayer([p1])
pNetwork1 = PerceptronNetwork([pLayer1_1, pLayer1_2])

# =============== NETWORK TEST [HALF ADDER] =============== #
pLayer2_1 = PerceptronLayer([p7, p2, p1])
pLayer2_2 = PerceptronLayer([p9, p8])
pNetwork2 = PerceptronNetwork([pLayer2_1, pLayer2_2])

# =============== PERCEPTRON TESTS =============== #
# print(f"\n{p1}")
# p1.activation([1, 1])
# p1.activation([1, 0])
# p1.activation([0, 1])
# p1.activation([0, 0])
#
# print(f"\n{p2}")
# p2.activation([1, 1])
# p2.activation([1, 0])
# p2.activation([0, 1])
# p2.activation([0, 0])
#
# print(f"\n{p3}")
# p3.activation([1])
# p3.activation([0])
#
# print(f"\n{p4}")
# p4.activation([1, 1, 1])
# p4.activation([1, 0, 1])
# p4.activation([0, 1, 0])
# p4.activation([1, 0, 0])
# p4.activation([0, 0, 1])
# p4.activation([1, 1, 0])
# p4.activation([0, 1, 1])
# p4.activation([0, 0, 0])
#
# print(f"\n{p5}")
# p5.activation([1, 1, 1])
# p5.activation([1, 0, 1])
# p5.activation([0, 1, 0])
# p5.activation([1, 0, 0])
# p5.activation([0, 0, 1])
# p5.activation([1, 1, 0])
# p5.activation([0, 1, 1])
# p5.activation([0, 0, 0])
#
# print(f"\n{p6}")
# p6.activation([1, 1])
# p6.activation([1, 0])
# p6.activation([0, 1])
# p6.activation([0, 0])
#
# print(f"\n{p7}")
# p7.activation([1, 1])
# p7.activation([1, 0])
# p7.activation([0, 1])
# p7.activation([0, 0])

# =============== NETWORK TESTS [XOR] =============== #
# pNetwork1.feed_forward([0, 0])
# pNetwork1.feed_forward([0, 1])
# pNetwork1.feed_forward([1, 0])
# pNetwork1.feed_forward([1, 1])

# =============== NETWORK TESTS [HALF ADDER] =============== #
pNetwork2.feed_forward([0, 0])
pNetwork2.feed_forward([0, 1])
pNetwork2.feed_forward([1, 0])
pNetwork2.feed_forward([1, 1])