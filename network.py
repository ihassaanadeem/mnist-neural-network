import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        # kaiming initialization compensates for 50% variance drop due to ReLU activation
        kaiming_w1 = np.sqrt(2 / self.input_size)
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * kaiming_w1
        self.b1 = np.zeros((1, self.hidden_size))
        
        kaiming_w2 = np.sqrt(2 / self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * kaiming_w2
        self.b2 = np.zeros((1, self.output_size))
        
        self.forward_prop(np.random.randn(self.batch_size, self.input_size))
    
    def forward_prop(self, data):
        d1 = np.dot(data, self.w1) + self.b1
        a1 = np.maximum(0, d1) #ReLU activation
        
        # logits
        d2 = np.dot(a1, self.w2) + self.b2
        
        # softmax activation function
        a2 = np.exp(d2)
        a2 = a2 / np.sum(a2, axis=1, keepdims=True)
        
        return a1, a2
    
    def loss_func(self, predictions, labels):
        # cross-entropy loss calc
        probs = predictions[np.arange(self.batch_size), labels]
        loss = -np.log(probs + 1e-15)
        
        # mean loss over batch
        avg_loss = np.mean(loss)
        
        return avg_loss
    
    def back_prop():
        return
    
    
nn = NeuralNetwork(784, 128, 10, 50)
# nn.loss_func(np.random.rand(50, 10), np.random.randint(0, 10, size=50))
print(1e-15)