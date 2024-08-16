
#Neural network to compute XOR function

import random
import math

learning_rate = 0.05

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sigm = sigmoid(x)
    return sigm*(1-sigm)
    

def compute_loss(y_real, y_pred):
    errors = []
    for i in range(0,len(y_real)):
        errors.append((y_real[i]-y_pred[i])**2)
    return sum(errors)/len(y_real)

def compute_derivative_loss(y_pred, y_real):
    return 2*(y_real - y_pred)



class Neuron():
    def __init__(self, previous_size) -> None:
        self.w = [random.uniform(-10,10) for _ in range(0,previous_size)]
        self.b = 0
        self.result = 0
        self.net = 0
        self.delta = 0
        self.last_input = 0
        
        
    def activate(self, x):
        self.last_input = x
        self.net = sum(w * xi for w, xi in zip(self.w, x)) + self.b  
        clipped_w_sum = max(min(self.net, 700), -700)
        result = sigmoid(clipped_w_sum)
        self.result = result
        return result
     
    
class Network():
    def __init__(self) -> None:
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def populate_layer(self, size, previous_size):
        layer = []
        for neuron_number in range(size):
            layer.append(Neuron(previous_size))
        self.layers.append(layer)
        return layer
    
    def feed(self, data):
        result = []
        for layer in self.layers:
            new_layer = []
            for neuron in layer:
                new_layer.append(neuron.activate(data))
            data = new_layer
            result = data
        return result
    
   
                
    def learn(self, x, y):
        y_pred = self.feed(x)
        y_real = y       
        for y_p, y_r, neuron in zip(y_pred, y_real, self.layers[-1]):
            dE_dy = compute_derivative_loss(y_r, y_p)
            dy_dnet = sigmoid_derivative(neuron.net)
            dE_dnet = dE_dy * dy_dnet
            neuron.delta = dE_dnet
            for index in range(0, len(neuron.w)):
                dnet_dw = neuron.last_input[index]
                de_dw = dE_dnet * dnet_dw
                neuron.w[index] -= learning_rate * de_dw
            neuron.b = neuron.b - learning_rate * dE_dnet
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            for j, neuron in enumerate(current_layer):
                dy_dnet = sigmoid_derivative(neuron.net)
                neuron.delta = sum(next_neuron.w[j]*next_neuron.delta for next_neuron in next_layer) * dy_dnet
                for k in range(len(neuron.w)):
                    neuron.w[k] -= learning_rate * neuron.delta * neuron.last_input[k]
                neuron.b -=  learning_rate * neuron.delta
            
                
    
net = Network()
layer_1 = net.populate_layer(2,2)
layer_2 = net.populate_layer(2,2)
layer_3 = net.populate_layer(1,2)

# XOR
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]


with open('xor_nn_output.txt', 'w') as file:
    epochs = 10000
    for epoch in range(epochs):
        total_loss = 0
        for x, y in training_data:
            net.learn(x, y)
            output = net.feed(x)
            total_loss += compute_loss(y, output)

        if epoch % 1000 == 0:
            file.write(f"Epoch {epoch}, Loss: {total_loss}\n")
            
            
    file.write("\nResults after training:\n")
    for inputs, target in training_data:
        output = net.feed(inputs)
        file.write(f"Input: {inputs}, Target: {target}, Output: {output}\n")