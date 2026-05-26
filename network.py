import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # kaiming initialization compensates for 50% variance drop due to ReLU activation
        kaiming_w1 = np.sqrt(2 / self.input_size)
        self.w1 = np.random.randn(self.input_size , self.hidden_size) * kaiming_w1
        self.b1 = np.zeros((1,self.hidden_size))
        
        kaiming_w2 = np.sqrt(2 / self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * kaiming_w2
        self.b2 = np.zeros((1, self.output_size))
        
    def forward_prop(self, data):
        
    
    def loss_func():
        pass
    
    def back_prop():
        pass
    
    