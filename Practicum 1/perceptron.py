
class Perceptron:
    def __init__(self, p_type, weights, b):
        """
        @param p_type: string
        @param weights: list
        @param b: int / float
        """
        self.p_type = p_type
        self.weights = weights
        self.b = b

    def execute_batch(self, inputs, expected):
        """
        ...
        @param inputs: list
        @param expected: list
        @return: list
        """
        total_outputs = []
        if len(inputs) == len(expected):
            for i in range(len(inputs)):
                output = self.activation(inputs[i])
                total_outputs.append(output)
                y_eval = output == expected[i]
                print(f"[{self.p_type}] | Input: {inputs[i]} | Output: {output} | Correct: {y_eval}")
        else:
            print("Length of inputs and expected are not equal.")

        return total_outputs

    def activation(self, inputs):
        """
        ...
        @param inputs: list
        @return: int
        """
        sum = self.b
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        if sum >= 0:
            return 1
        else:
            return 0

    def __str__(self):
        """
        Function to make the perceptron object printable for informational purposes.
        @return: string
        """
        return f"Perceptron | Type: '{self.p_type}' | Weights: '{self.weights}' | Bias: '{self.b}'"


class PerceptronLayer:
    def __init__(self, p_type, ptrons):
        """
        @param p_type: string
        @param ptrons: list
        """
        self.p_type = p_type
        self.ptrons = ptrons

    def activation(self, inputs):
        """
        ...
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
        return f"Perceptron Layer | Amount of Perceptrons: '{len(self.ptrons)}' | Perceptron types: '{self.p_type}'"


class PerceptronNetwork:
    def __init__(self, net_type, pLayers):
        """
        @param net_type: string
        @param pLayers: list
        """
        self.net_type = net_type
        self.pLayers = pLayers

    def feed_forward(self, inputs, expected):
        """
        ...
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
                y_eval = output_value == expected[i]
                print(f"[{self.net_type}] | Input: {inputs[i]} | Output: {output_value} | Correct: {y_eval}")
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
