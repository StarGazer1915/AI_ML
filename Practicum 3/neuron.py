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

    def execute_batch(self, inputs, expected):
        """
        The main function to run multiple activations.
        The function runs an activation for each input given and generates an output. The output is then
        compared and put into a print statement to give an accurate visual representation of the output
        and if it's correct/expected. All the outputs are then returned as a list.
        @param inputs: list
        @param expected: list
        @return: list
        """
        total_outputs = []
        if len(inputs) == len(expected):
            for i in range(len(inputs)):
                output = self.activation(inputs[i])
                total_outputs.append(output)
                print(f"[{self.n_type}] | Input: {inputs[i]} | Output: {output} | Expected: {expected[i]}")
        else:
            print("Length of inputs and expected are not equal.")

        return total_outputs

    def activation(self, inputs):
        """
        This function handles the activation of the neuron.
        The function uses the sigmoid function to determine the output.
        @param inputs: list
        @return: int
        """
        sum = self.b
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        return self.sigmoid(sum)

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
        return f"Neuron | Type: '{self.n_type}' | Weights: '{self.weights}' | Bias: '{self.b}'"


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

    def feed_forward(self, inputs, expected):
        """
        The main function to run multiple activations in multiple layers.
        The function runs an activation for each input given and generates an output. The output is then
        compared and put into a print statement to give an accurate visual representation of the output
        and if it's correct/expected. All the outputs are then returned as a list.
        @param inputs: list
        @param expected: list
        @return: list
        """
        total_outputs = []
        if len(inputs) == len(expected):
            for i in range(len(inputs)):
                output_value = inputs[i]
                for layer in self.nLayers:
                    output_value = layer.activation(output_value)
                total_outputs.append(output_value)
                print(f"[{self.net_type}] | Input: {inputs[i]} | Output: {output_value} | Expected: {expected[i]}")
        else:
            print("Length of inputs and expected are not equal.")

        return total_outputs

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
