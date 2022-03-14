import math


class Neuron:
    def __init__(self, n_type, weights, b):
        """
        @param n_type: string
        @param weights: list
        @param b: int / float
        """
        self.n_type = n_type
        self.weights = weights
        self.b = b
        self.error = 0
        self.gradient = 0
        self.new_weights = []
        self.new_bias = 0
        self.fwd_err_lst = []
        self.prev_inputs = []

    def activation(self, inputs):
        """
        This function handles the activation of the neuron.
        The function uses the sigmoid function to determine the output.
        @param inputs: list
        @return: int
        """
        self.prev_inputs = inputs
        som = self.b
        for i in range(len(inputs)):
            som += inputs[i] * self.weights[i]

        output = self.sigmoid(som)

        return output

    def calc_new_weights_and_bias(self, lr):
        """"""
        new_w = []
        for i in range(len(self.weights)):
            new_w.append(self.weights[i] - lr * self.prev_inputs[i] * self.error)

        self.new_weights = new_w
        self.new_bias = self.b - (lr * self.error)
        return

    def update(self):
        """
        Updates the weights and bias with the new values.
        @return: void
        """
        self.weights = self.new_weights
        self.b = self.new_bias
        return

    def calc_error_output_neuron(self, output, expected):
        """"""
        self.error = output * (1 - output) * -(expected - output)

    def calc_error_hidden_neuron(self, output):
        """
        Calculates the hidden error of the previous layer.
        @param output: float
        @return: float
        """
        der_output = output * (1 - output)
        som = 0
        for i in range(len(self.weights)):
            som += self.weights[i] * self.fwd_err_lst[i]

        return der_output * som


    def calc_gradient(self, output):
        """"""
        self.gradient = self.error * output

    def sigmoid(self, z):
        """
        Applies the sigmoid function.
        @param z: int / float
        @return: int / float
        """
        return 1 / (1 + math.e**(-z))

    def __str__(self):
        """
        Function to make the neuron object printable for informational purposes.
        @return: string
        """
        return f"Neuron | Type: '{self.n_type}' | Weights: '{self.weights}' | Bias: '{self.b}' | Error: '{self.error}'"


class NeuronLayer:
    def __init__(self, n_type, neurons):
        """
        @param n_type: string
        @param neurons: list
        """
        self.n_type = n_type
        self.neurons = neurons

    def activation(self, inputs):
        """
        This function handles the activation of the neurons that are in the layer.
        This function uses the sigmoid function from the activation() function in the Neuron class.
        @param inputs: list
        @return: list
        """
        output = []
        for neuron in self.neurons:
            output.append(neuron.activation(inputs))

        return output

    def __str__(self):
        """
        Function to make the Neuron layer object printable for informational purposes.
        @return: string
        """
        return f"Neuron Layer | Amount of Neurons: '{len(self.neurons)}' | Neuron types: '{self.n_type}'"


class NeuronNetwork:
    def __init__(self, net_type, nLayers):
        """
        @param net_type: string
        @param nLayers: list
        """
        self.net_type = net_type
        self.nLayers = nLayers

    def train(self, lr, inputs, expected, num_of_epochs):
        """"""
        for epoch in range(1, num_of_epochs + 1):
            self.feed_forward(lr, inputs, expected)
            self.backpropagation()

    def backpropagation(self):
        # ...
        return


    def feed_forward(self, lr, inputs, expected):
        """
        The main function to run multiple activations in multiple layers.
        The function runs an activation for each input given and generates an output. The output is then
        compared and put into a print statement to give an accurate visual representation of the output
        and if it's correct/expected. All the outputs are then returned as a list.
        @param inputs: list
        @param expected: list
        @return: list
        """
        output_value = []
        if len(inputs) == len(expected):
            for i in range(len(inputs)):
                output_value = inputs[i]
                for layer in self.nLayers:
                    output_value = layer.activation(output_value)
                print(f"[{self.net_type}] | Input: {inputs[i]} | Output: {output_value} | Expected: {expected[i]}")
                print()
                for n in range(len(self.nLayers[-1].neurons)):
                    neuron = self.nLayers[-1].neurons[n]
                    neuron.calc_error_output_neuron(output_value[n], expected[i][n])
                    neuron.calc_new_weights_and_bias(lr)
        else:
            print("Length of inputs and expected are not equal.")
        return output_value

    def get_layers(self):
        """
        Function that returns the layers the network holds.
        @return: list
        """
        return self.nLayers

    def __str__(self):
        """
        Function to make the Neuron Network printable for informational purposes.
        @return: string
        """
        return f"Neuron Network | Type: '{self.net_type}' | Amount of Layers: '{len(self.nLayers)}'"
