
class Perceptron:
    def __init__(self, p_type, weights, b):
        self.p_type = p_type
        self.weights = weights
        self.b = b

    def execute_batch(self, inputs, expected):
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
        sum = self.b
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        if sum >= 0:
            return 1
        else:
            return 0

    def __str__(self):
        return f"Perceptron | Type: '{self.p_type}' | Weights: '{self.weights}' | Bias: '{self.b}'"


class PerceptronLayer:
    def __init__(self, p_type, ptrons):
        self.p_type = p_type
        self.ptrons = ptrons

    def activation(self, inputs):
        output = []
        for perceptron in self.ptrons:
            output.append(perceptron.activation(inputs))

        return output

    def __str__(self):
        return f"Perceptron Layer | Amount of Perceptrons: '{len(self.ptrons)}' | Perceptron types: '{self.p_type}'"


class PerceptronNetwork:
    def __init__(self, net_type, pLayers):
        self.net_type = net_type
        self.pLayers = pLayers

    def feed_forward(self, inputs, expected):
        output_value = []
        if len(inputs) == len(expected):
            for i in range(len(inputs)):
                output_value = inputs[i]
                for layer in self.pLayers:
                    output_value = layer.activation(output_value)
                y_eval = output_value == expected[i]
                print(f"[{self.net_type}] | Input: {inputs[i]} | Output: {output_value} | Correct: {y_eval}")
        else:
            print("Length of inputs and expected are not equal.")

        return output_value

    def get_layers(self):
        return self.pLayers

    def __str__(self):
        return f"Perceptron Network | Type: '{self.net_type}' | Amount of Layers: '{len(self.pLayers)}'"