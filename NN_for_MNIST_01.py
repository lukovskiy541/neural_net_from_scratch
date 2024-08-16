
#Neural network to classify from MNIST dataset 0 and 1 
from loading_dataset import data, test_data
import random
import math

learning_rate = 0.1

def sigmoid(x):
    clipped_x = max(min(x, 700), -700)
    return 1 / (1 + math.exp(-clipped_x))

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
        self.w = [random.uniform(-0.5,0.5) for _ in range(0,previous_size)]
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
layer_1 = net.populate_layer(128, 784)
layer_2 = net.populate_layer(64, 128)
layer_3 = net.populate_layer(1, 64)



training_data = data

with open('MNIST_01_nn_output.txt', 'w') as file:
    file.write("Learning started")
    epochs = 200
    last_loss = 100
    for epoch in range(epochs):
        total_loss = 0
        
        for x, y in training_data:
        
            net.learn(x, y)

            output = net.feed(x)
        
            total_loss += compute_loss(y, output)
    
    
        if epoch % 20== 0:
            file.write(f"Epoch {epoch}, Loss: {total_loss}\n")
            file.write(f"{learning_rate}")
                
                
    file.write("\nResults after training:\n")
    wrong_predictions = 0
    for inputs, target in test_data:
        output = net.feed(inputs)
        output = 0 if output[0] < 0.2 else 1
        if target[0] != output:
            wrong_predictions += 1
        file.write(f"Target: {target}, Output: {output}\n")
    file.write(f"Count of wrong predictions: {wrong_predictions}/{len(training_data)}")