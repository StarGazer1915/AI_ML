
class Perceptron:
    def __init__(self, p_type, weights, b):
        self.p_type = p_type
        self.weights = weights
        self.b = b

    def activation(self, inputs):
        sum = self.b
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        if sum >= 0:
            print(f"[{self.p_type}] | Inputs: {inputs} | Output = 1")
            return 1
        else:
            print(f"[{self.p_type}] | Inputs: {inputs} | Output = 0")
            return 0

    def __str__(self):
        return f"[{self.p_type}] Perceptron | weights: '{self.weights}' | bias: '{self.b}' "


class PerceptronLayer:
    def __init__(self, ptrons):
        self.ptrons = ptrons

    def activation(self, inputs):
        output = []
        for perceptron in self.ptrons:
            output.append(perceptron.activation(inputs))

        print(f"----- New input = {output} -----\n")
        return output


class PerceptronNetwork:
    def __init__(self, pLayers):
        self.pLayers = pLayers

    def feed_forward(self, inputs):
        output_values = inputs
        for layer in self.pLayers:
            output_values = layer.activation(output_values)

        print(f"=================================================\n"
              f"| Input was: {inputs} | Network output is: {output_values} |\n"
              f"=================================================\n")
        return output_values