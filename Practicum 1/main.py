from perceptron import Perceptron, PerceptronLayer, PerceptronNetwork

# =============== DEFINING POSSIBILITIES =============== #
pos_1x = [[1], [0]]
pos_2x = [[1, 1], [1, 0], [0, 1], [0, 0]]
pos_3x = [[1, 1, 1], [0, 1, 1], [1, 1, 0],
          [0, 0, 1], [0, 1, 0], [1, 0, 0],
          [1, 0, 1], [0, 0, 0]]

# =============== DEFINING PERCEPTRONS =============== #
p1 = Perceptron("AND", [0.5, 0.5], -1)
p2 = Perceptron("OR", [0.1, 0.1], -0.1)
p3 = Perceptron("NOT", [-1], 0.5)
p4 = Perceptron("Figuur 2.8", [0.6, 0.3, 0.2], -0.4)
p5 = Perceptron("NOR 3x", [-1, -1, -1], 0)
p6 = Perceptron("NOR 2x", [-1, -1], 0)
p7 = Perceptron("NAND", [-0.1, -0.1], 0.1)
p8 = Perceptron("PASSTROUGH", [0, 0, 1], -0.1)
p9 = Perceptron("HA-AND", [0.5, 0.5, 0], -1)

# =============== DEFINING NETWORK [XOR] =============== #
pLayer1_1 = PerceptronLayer('NAND, OR', [p7, p2])
pLayer1_2 = PerceptronLayer('AND', [p1])
pNetwork1 = PerceptronNetwork('XOR', [pLayer1_1, pLayer1_2])

# =============== DEFINING NETWORK [HALF ADDER] =============== #
pLayer2_1 = PerceptronLayer('NAND, OR, AND', [p7, p2, p1])
pLayer2_2 = PerceptronLayer('HA-AND, PASSTROUGH', [p9, p8])
pNetwork2 = PerceptronNetwork('HALF ADDER', [pLayer2_1, pLayer2_2])


# ===============  TEST FUNCTIONS =============== #
def test_perceptor(p, inputs, expected):
    """
    This function tests the perceptron by comparing the output and expected values.
    The print statements and returned output from the 'execute_batch' method will show
    if the outputs are what is expected from the perceptor.
    @param p: perceptor object
    @param inputs: list
    @param expected: list
    @return: list
    """
    print(f"\n{p}")
    return p.execute_batch(inputs, expected)


def test_network(net, inputs, expected):
    """
    Mostly same as the test_perceptor() function but for a network.
    The function also prints the layers of the network to visualize it's structure.
    @param net: network object
    @param inputs: list
    @param expected: list
    @return: list
    """
    print(f"\n{net}")
    for layer in net.get_layers():
        print(layer)
    return net.feed_forward(inputs, expected)


# =============== TEST CASES PERCEPTORS =============== #
test_perceptor(p1, pos_2x, [1, 0, 0, 0])
test_perceptor(p2, pos_2x, [1, 1, 1, 0])
test_perceptor(p3, pos_1x, [0, 1])
test_perceptor(p4, pos_3x, [1, 1, 1, 0, 0, 1, 1, 0])
test_perceptor(p5, pos_3x, [0, 0, 0, 0, 0, 0, 0, 1])
test_perceptor(p6, pos_2x, [0, 0, 0, 1])
test_perceptor(p7, pos_2x, [0, 1, 1, 1])
test_perceptor(p8, pos_3x, [1, 1, 0, 1, 0, 0, 1, 0])
test_perceptor(p9, pos_3x, [1, 0, 1, 0, 0, 0, 0, 0])

# =============== TEST CASES NETWORKS =============== #
test_network(pNetwork1, pos_2x, [[0], [1], [1], [0]])
test_network(pNetwork2, pos_2x, [[0, 1], [1, 0], [1, 0], [0, 0]])
