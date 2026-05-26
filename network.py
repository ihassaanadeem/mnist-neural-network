import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # kaiming initialization compensates for 50% variance drop due to ReLU activation
        kaiming_w1 = np.sqrt(2 / self.input_size)
        self.w1 = np.random.randn(self.hidden_size, self.input_size) * kaiming_w1
        self.b1 = np.zeros((self.hidden_size, 1))
        
        kaiming_w2 = np.sqrt(2 / self.hidden_size)
        self.w2 = np.random.randn(self.output_size, self.hidden_size) * kaiming_w2
        self.b2 = np.zeros((self.output_size, 1))
        
        self.forward_prop(np.random.randn(self.input_size, 1))
    
    def forward_prop(self, data):
        d1 = np.dot(self.w1, data) + self.b1
        a1 = np.maximum(0, d1) #ReLU activation
        
        # logits
        d2 = np.dot(self.w2, a1) + self.b2
        
        # softmax activation function
        a2 = np.exp(d2)
        a2 = a2 / np.sum(a2)
        
        # print(a2.shape)
        
        return a1, a2
    
    def loss_func():
        return
    
    def back_prop():
        return
    
    
nn = NeuralNetwork(784, 128, 10)