import math

class Neuron:
    def __init__(self, n_type, weights, b):
        """
        @param p_type: string
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
        This function handles the activation of the perceptron.
        The function uses the sigmoid function to determine the output.
        @param inputs: list
        @return: int
        """
        sum = self.b
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        return self.sigmoid(sum)

    def sigmoid(self, z):
        return 1 / (1 + math.e**(-z))

    def __str__(self):
        """
        Function to make the perceptron object printable for informational purposes.
        @return: string
        """
        return f"Neuron | Type: '{self.n_type}' | Weights: '{self.weights}' | Bias: '{self.b}'"


class NeuronLayer:
    def __init__(self, n_type, ptrons):
        """
        @param p_type: string
        @param ptrons: list
        """
        self.n_type = n_type
        self.ptrons = ptrons

    def activation(self, inputs):
        """
        This function handles the activation of the perceptrons that are in the layer.
        This function uses the step function from the activation() function in the perceptron class.
        @param inputs: list
        @return: list
        """
        output = []
        for perceptron in self.ptrons:
            output.append(perceptron.activation(inputs))

        return output

    def __str__(self):
        """
        Function to make the perceptron layer object printable for informational purposes.
        @return: string
        """
        return f"Perceptron Layer | Amount of Perceptrons: '{len(self.ptrons)}' | Perceptron types: '{self.n_type}'"


class NeuronNetwork:
    def __init__(self, net_type, pLayers):
        """
        @param net_type: string
        @param pLayers: list
        """
        self.net_type = net_type
        self.pLayers = pLayers

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
                for layer in self.pLayers:
                    output_value = layer.activation(output_value)
                total_outputs.append(output_value)
                correct = output_value == expected[i]
                print(f"[{self.net_type}] | Input: {inputs[i]} | Output: {output_value} | Correct: {correct}")
        else:
            print("Length of inputs and expected are not equal.")

        return total_outputs

    def get_layers(self):
        """
        Function that returns the layers the network holds.
        @return: list
        """
        return self.pLayers

    def __str__(self):
        """
        Function to make the perceptron network printable for informational purposes.
        @return: string
        """
        return f"Perceptron Network | Type: '{self.net_type}' | Amount of Layers: '{len(self.pLayers)}'"
