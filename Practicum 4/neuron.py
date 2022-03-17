import math
from random import normalvariate


class Neuron:
    def __init__(self, n_type, num_weights):
        """
        @param n_type: string
        @param weights: list
        @param b: int / float
        """
        self.n_type = n_type
        self.weights = [normalvariate(0, 0.1) for i in range(num_weights)]
        self.b = normalvariate(0, 0.1)
        self.error = 0
        self.prev_n_weights = []
        self.prev_inputs = []
        self.prev_output = 0
        self.new_weights = []
        self.new_bias = 0

    def activation(self, inputs):
        """
        This function handles the activation of the neuron. The function uses the sigmoid function to
        determine the output. It also stores the most recent inputs and outputs for use in backpropagation.
        @param inputs: list
        @return: int
        """
        self.prev_inputs = inputs
        som = self.b
        for i in range(len(inputs)):
            som += inputs[i] * self.weights[i]

        self.prev_output = self.sigmoid(som)
        return self.prev_output

    def calc_new_weights_and_bias(self, lr):
        """
        Calculates the new weights and bias for the neuron. It calculates the deltas and then applies them
        to the weights. The new weights and bias are then temporarily stored until the real values are
        updated all at once after backpropagation.
        @param lr: int / float
        @return: void
        """
        new_w = []
        for i in range(len(self.weights)):
            new_weight = self.weights[i] - (lr * (self.prev_inputs[i] * self.error))
            new_w.append(new_weight)

        self.new_weights = new_w
        self.new_bias = self.b - (lr * self.error)
        return

    def calc_error_output_neuron(self, output, expected):
        """
        Calculates the error of the current neuron. This function is only used to calculate
        the errors of the output neurons which can then be used in backpropagation.
        @param output: int / float
        @param expected: int / float
        @return: void
        """
        der_output = output * (1 - output)
        self.error = der_output * -(expected - output)
        return

    def calc_error_hidden_neuron(self, indx, fwd_w_lst, fwd_err_lst):
        """
        Calculates the error of a hidden neuron. It uses the weights and errors of the previous
        layer to calculate the hidden error so that it's delta's can be calculated (for example).
        @param indx: int
        @param fwd_w_lst: list
        @param fwd_err_lst: list
        @return: void
        """
        der_output = self.prev_output * (1 - self.prev_output)
        som = 0
        for i in range(len(fwd_w_lst)):
            som += fwd_w_lst[i][indx] * fwd_err_lst[i]

        self.error = der_output * som
        return

    def update(self):
        """
        Updates the weights and bias with the new values.
        @return: void
        """
        self.weights = self.new_weights
        self.b = self.new_bias
        return

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
        """
        This function runs the feed_forward() function which contains the function calls of the backpropagation()
        and update_all() functions for the amount of epochs given. It will then store the output and will use
        the final output in a print statement to show the results of the training. If an lr of 0 is given then
        this function transforms into a test function as we simply run the neural network with inputs and don't
        update the weights and bias. In short, this function trains the neural network and shows the final results.
        @param lr: int / float
        @param inputs: list
        @param expected: list
        @param num_of_epochs: int
        @return: list
        """
        output = []
        print("\nWORKING...")
        for epoch in range(1, num_of_epochs + 1):
            output = self.feed_forward(lr, inputs, expected)

        print(f"\n| =============== EPOCH {num_of_epochs} =============== | MSE: {output[1]}")
        for i in range(len(output[0])):
            print(f"Inputs: {inputs[i]} | Output: {output[0][i]} | Expected: {expected[i]}\n")

        return output

    def feed_forward(self, lr, inputs, expected):
        """
        The main function to run multiple activations in multiple layers.
        The function runs an activation for each input given and generates an output. The output is then
        passed on and put into a print statement to give an accurate visual representation of the output
        and if it's expected. When an input has passed trough the network and an output value is generated
        the output neurons then calculate their errors and new weights/bias. Should the network contain
        more than 1 layer the backpropagation process is run. The update_all() function then
        updates all neurons in the network with their new (already stored) weights and biases.
        All the outputs of all inputs are then returned as a list.
        @param lr: float
        @param inputs: list
        @param expected: list
        @return: list ([[outputs], MSE])
        """
        total_output = []
        if len(inputs) == len(expected):
            for i in range(len(inputs)):
                output_value = inputs[i]
                for layer in self.nLayers:
                    output_value = layer.activation(output_value)
                total_output.append(output_value)

                for n in range(0, len(self.nLayers[-1].neurons)):
                    neuron = self.nLayers[-1].neurons[n]
                    neuron.calc_error_output_neuron(output_value[n], expected[i][n])
                    neuron.calc_new_weights_and_bias(lr)

                if len(self.nLayers) > 1:
                    self.backpropagation(lr)

                self.update_all()
        else:
            print("Length of inputs and expected are not equal.")

        return [total_output, self.calc_epoch_MSE(total_output, expected)]

    def backpropagation(self, lr):
        """
        This function generates the hidden errors and new weights/biases for the hidden neurons
        in the network. It first gathers the errors and weights from the previous layer (in front of
        the current layer, like the output layer for example) and stores these values in lists.
        Then for each neuron in all hidden layers and each neuron in that layer the hidden errors
        and new weights/biases are calculated and stored.
        @param lr: int
        @return: void
        """
        length = len(self.nLayers)-2
        for i in range(length, -1, -1):
            prev_err = []
            prev_weights = []
            for prev_neuron in self.nLayers[i+1].neurons:
                prev_err.append(prev_neuron.error)
                prev_weights.append(prev_neuron.weights)
            count = 0
            for neuron in self.nLayers[i].neurons:
                neuron.calc_error_hidden_neuron(count, prev_weights, prev_err)
                neuron.calc_new_weights_and_bias(lr)
                count += 1
        return

    def update_all(self):
        """
        This function simply goes trough all layers and updates
        the neurons with their new weights/biases.
        @return: void
        """
        for layer in self.nLayers:
            for n in layer.neurons:
                n.update()
        return

    def calc_epoch_MSE(self, outputs, expected):
        """
        Calculates the Mean Squared Error for the current epoch.
        @param outputs: list
        @param expected: list
        @return: float
        """
        total_loss = 0
        for i in range(len(outputs)):
            som = 0
            for j in range(len(outputs[i])):
                som += (expected[i][j] - outputs[i][j])**2
            total_loss += 0.5 * som

        return total_loss / len(outputs)

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
