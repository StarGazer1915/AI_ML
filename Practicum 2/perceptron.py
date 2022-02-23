
class Perceptron:
    def __init__(self, p_type, weights, b, learning_rate):
        """
        @param p_type: string
        @param weights: list
        @param b: int / float
        @param learning_rate: int / float
        """
        self.p_type = p_type
        self.weights = weights
        self.b = b
        self.lr = learning_rate
        self.t_loss = 0
        self.loss_count = 0

    def train(self, inputs, expected, num_of_epochs):
        """
        This function starts the training process. For each epoch the inputs are put into
        the activation function, then the output is checked with the expected values and printed.
        Then the update function is called to update the parameters that are not outputting the right
        values. At the end of each epoch the loss function is called to calculate the total loss (MSE).
        @param inputs: list
        @param expected: list
        @param num_of_epochs: int
        """
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
        """
        This function handles the activation of the perceptron.
        The function uses the step function to determine the output. ( >= 0 )
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

    def update(self, d, y, inputs):
        """
        This function updates the weights and bias with new values based on
        the output and expected value. A delta is calculated for the weights and
        bias and is then applied to the existing value, updating it.
        @param d: int
        @param y: int
        @param inputs: list
        """
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
        """
        Calculates the total loss (MSE) of an training epoch.
        (MSE = (sum |d - y|^2) / n)
        @return: float
        """
        print(f"[{self.p_type}] | Total loss (MSE): {self.t_loss / self.loss_count}\n")
        return self.t_loss / self.loss_count

    def __str__(self):
        """
        This function makes the object printable and shows usefull information about the object.
        @return: string
        """
        return f"Perceptron | Type: '{self.p_type}' | Weights: '{self.weights}' | Bias: '{self.b}' | " \
               f"Learning Rate: '{self.lr}'"
