
class Perceptron:
    def __init__(self, p_type, weights, b, learning_rate):
        self.p_type = p_type
        self.weights = weights
        self.b = b
        self.lr = learning_rate
        self.t_loss = 0
        self.loss_count = 0

    def train(self, inputs, expected, num_of_epochs):
        if len(inputs) == len(expected):
            for epoch in range(1, num_of_epochs+1):
                self.t_loss = 0
                self.loss_count = 0
                print(f"------------------ Epoch: {epoch} ------------------")
                for i in range(len(inputs)):
                    output = self.activation(inputs[i])
                    y_eval = output == expected[i]
                    print(f"[{self.p_type}] | Input: {inputs[i]} | Output: {output} | Correct: {y_eval}")
                    self.update(expected[i], output, inputs[i])
                self.loss()
                if self.t_loss / self.loss_count == 0:
                    break
        else:
            print("Length of inputs and expected are not equal.")

    def activation(self, inputs):
        sum = self.b
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        if sum >= 0:
            return 1
        else:
            return 0


    def update(self, d, y, inputs):
        new_weights = []
        for i in range(len(self.weights)):
            w_delta = self.lr * ((d - y) * inputs[i])
            new_weights.append(self.weights[i] + w_delta)
            self.t_loss += (d - y)**2
            self.loss_count += 1

        b_delta = self.lr * (d - y)

        self.b += b_delta
        self.weights = new_weights

    def loss(self):
        print(f"[{self.p_type}] | Total loss (MSE): {self.t_loss / self.loss_count}\n")
        return self.t_loss / self.loss_count

    def __str__(self):
        return f"Perceptron | Type: '{self.p_type}' | Weights: '{self.weights}' | Bias: '{self.b}' | " \
               f"Learning Rate: '{self.lr}'"


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