import random
import math

learning_rate = 0.05

def sigmoid(x):
    return 1 / (1 + math.exp(x))

def sigmoid_derivative(self,x):
        return sigmoid(x)*(1-sigmoid(x))
    

def compute_loss(y_real, y_pred):
    errors = []
    for i in range(0,len(y_real)):
        errors.append((y_real[i]-y_pred[i])**2)
    return sum(errors)/len(y_real)

def compute_derivative_loss(y_pred, y_real):
    errors = []
    for i in range(0,len(y_real)):
        errors.append(2*(y_real[i]-y_pred[i]))
    return 2*sum(errors)/len(y_real)


class Neuron():
    def __init__(self, previous_size) -> None:
        self.w = [random.uniform(-10,10) for _ in range(0,previous_size)]
        self.b = 0
        self.result = 0
        self.z = 0
        self.de_dy_j___x___dy_j_dnet_j = 0
        
        
    def activate(self, x):
        self.z = sum(w * xi for w, xi in zip(self.w, x)) + self.b  
        clipped_w_sum = max(min(self.z, 700), -700)
        result = sigmoid(-clipped_w_sum)
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
    
    def back_prop(self,loss,y_pred,y_real):
        for neuron_number in range(0, len(result)):
            neuron = self.layers[-1][neuron_number]
            de_dy_j = compute_derivative_loss(y_real[neuron_number], y_pred[neuron_number])
            dy_j_dnet_j = sigmoid_derivative(neuron.z)
            gradient_for_1_neuron = []
            
            for input in neuron.inputs:
               
                gradient_for_1_neuron.append(de_dy_j * dy_j_dnet_j * input)
            neuron.de_dy_j___x___dy_j_dnet_j = de_dy_j * dy_j_dnet_j
            for i in range(0,len(neuron.w)):
                neuron.w[i] = neuron.w[i] - learning_rate*gradient_for_1_neuron[i]
            neuron.b = neuron.b - learning_rate* de_dy_j * dy_j_dnet_j
        layer_index = -2
        for layer in reversed(self.layers):
            if self.layers.index(layer) == 0:
                for neuron in layer:
                    for i in range(0,len(neuron.w)):
                        neuron.w[i] = neuron.w[i] - learning_rate*gradient_for_1_neuron[i]
            for neuron in self.layers[layer_index]:
                gradient = 0
                len_of_next_layer = len(self.layers[layer_index+1])
                neuron_index = 0
                for neuron in self.layers[layer_index+1]:
                    gradient = gradient + neuron.de_dy_j___x___dy_j_dnet_j * neuron.w[neuron_index]
                    neuron_index = neuron_index + 1   
                d_yh_dnet_h = sigmoid_derivative(neuron.z)
                h_grad = d_yh_dnet_h * gradient
                neuron.de_dy_j___x___dy_j_dnet_j = h_grad
                
    def learn(self, x, y):
        y_pred = self.feed(data)
        y_real = y       
        for y_p, y_r, neuron in zip(y_pred, y_real, self.layers[-1]):
            dE_dy = compute_derivative_loss(y_r, y_p)
            dy_dnet = sigmoid_derivative()
        
            
            
                
        
        
net = Network()

layer_1 = net.populate_layer(3,3)
layer_2 = net.populate_layer(3,3)
layer_3 = net.populate_layer(2,2)

for i in layer_1:
    print(i.w,i.b)

for i in layer_2:
    print(i.w,i.b)
    

data = [6.3, 1.2, 4.5]
data2 = [4.5, 6.3, 1.2]
data3 = [1.2, 6.3, 4.5]
results = [0.5, 0.7, 0.11]
test = [ 6.3, 4.5, 1.2]



result = net.feed(data)

print(result)

