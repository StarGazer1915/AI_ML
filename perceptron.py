
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
            print(f"Inputs: {inputs} | Output = 1")
            return 1
        else:
            print(f"Inputs: {inputs} | Output = 0")
            return 0

    def __str__(self):
        return f"[{self.p_type}] Perceptron | weights: '{self.weights}' | bias: '{self.b}' "
